package com.nivukx.nox.core.network

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import com.nivukx.nox.core.model.DeviceId
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.net.SocketException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/**
 * Production-bounded UDP transport for reachable remote NoX camera nodes.
 *
 * This adapter deliberately stays below the distributed route and evidence layers:
 * - SecureRemoteEvidenceCodec authenticates the complete logical NoX packet;
 * - RemoteDatagramFragmentCodec provides MTU-safe fragmentation;
 * - RemoteDatagramReassembler bounds partial-message memory;
 * - PacketReplayGuard rejects duplicate logical packets.
 *
 * It supports static reachable endpoints over IPv4 or IPv6. It does not claim NAT traversal,
 * rendezvous, TURN, or internet discovery. Those are separate infrastructure concerns.
 */
class UdpRemoteEvidenceTransport(
    private val context: Context,
    private val config: Config = Config(),
    private val securityKey: ByteArray,
) : AutoCloseable {

    data class Config(
        val bindPort: Int = 9100,
        val maximumDatagramBytes: Int =
            RemoteDatagramFragmentCodec.MAX_DATAGRAM_BYTES,
        val receiveBufferBytes: Int = 8 * 1024,
        val receiveTimeoutMillis: Int = 100,
        val maximumPacketBytes: Int =
            NoXWireProtocol.MAX_PACKET_BYTES,
        val maximumInFlightAssemblies: Int = 128,
        val maximumInFlightBytes: Int = 32 * 1024 * 1024,
        val maximumAssemblyAgeNanos: Long = 500_000_000L,
        val maximumPeers: Int = 256,
        val maximumDecodeWorkers: Int =
            minOf(
                8,
                Runtime.getRuntime()
                    .availableProcessors()
                    .coerceAtLeast(1),
            ),
        val maximumDecodeQueue: Int = 1024,
        val decodeFreshnessPriorityEnabled: Boolean = true,
    ) {
        init {
            require(bindPort in 1..65_535)
            require(
                maximumDatagramBytes in
                    576..RemoteDatagramFragmentCodec.MAX_DATAGRAM_BYTES,
            )
            require(receiveBufferBytes >= 2048)
            require(receiveTimeoutMillis in 10..5_000)
            require(
                maximumPacketBytes in
                    4 * 1024..NoXWireProtocol.MAX_PACKET_BYTES,
            )
            require(maximumInFlightAssemblies in 1..1024)
            require(maximumInFlightBytes in 64 * 1024..256 * 1024 * 1024)
            require(maximumAssemblyAgeNanos > 0L)
            require(maximumPeers in 1..1024)
            require(maximumDecodeWorkers in 1..16)
            require(maximumDecodeQueue in 64..16_384)
        }
    }

    data class Endpoint(
        val host: String,
        val port: Int,
    ) {
        init {
            require(host.isNotBlank())
            require(port in 1..65_535)
            require(!host.any { it.isWhitespace() })
        }

        val address: InetSocketAddress
            get() = InetSocketAddress(host, port)

        override fun toString(): String =
            host + ":" + port
    }

    data class Received(
        val source: InetSocketAddress,
        val securePacket: ByteArray,
        val decoded:
            SecureRemoteEvidenceCodec.Decoded,
    )

    data class Diagnostics(
        val running: Boolean,
        val bindPort: Int,
        val receivedDatagrams: Long,
        val droppedDatagrams: Long,
        val decodedPackets: Long,
        val authenticationFailures: Long,
        val replayRejectedPackets: Long,
        val inFlightAssemblies: Int,
        val inFlightBytes: Int,
        val decodeQueueDepth: Int = 0,
        val decodeOutstandingTasks: Int = 0,
        val permissionGranted: Boolean,
    )

    private val _received =
        MutableSharedFlow<Received>(
            extraBufferCapacity = 128,
        )

    val received: SharedFlow<Received> =
        _received.asSharedFlow()

    private val executor: ExecutorService =
        Executors.newSingleThreadExecutor { runnable ->
            Thread(
                runnable,
                "nox-udp-rx",
            ).apply {
                isDaemon = true
            }
        }

    private val decodeExecutor =
        BoundedTimestampPriorityExecutor(
            workerCount = config.maximumDecodeWorkers,
            maximumOutstandingTasks = config.maximumDecodeQueue,
            threadName = "nox-udp-decode",
        )

    private val reassembler =
        RemoteDatagramReassembler(
            RemoteDatagramReassembler.Config(
                maximumInFlightAssemblies =
                    config.maximumInFlightAssemblies,
                maximumInFlightBytes =
                    config.maximumInFlightBytes,
                maximumAssemblyAgeNanos =
                    config.maximumAssemblyAgeNanos,
            ),
        )

    private val replayGuard =
        PacketReplayGuard(
            maximumPeers = config.maximumPeers,
        )

    private val sentDatagrams =
        AtomicLong(0L)
    private val receivedDatagrams =
        AtomicLong(0L)
    private val droppedDatagrams =
        AtomicLong(0L)
    private val decodedPackets =
        AtomicLong(0L)
    private val authenticationFailures =
        AtomicLong(0L)

    private val peers =
        ConcurrentHashMap<Endpoint, InetSocketAddress>()

    @Volatile
    private var socket: DatagramSocket? = null

    @Volatile
    private var running = false

    fun permissionGranted(): Boolean =
        if (Build.VERSION.SDK_INT >= 37) {
            context.checkSelfPermission(
                Manifest.permission.ACCESS_LOCAL_NETWORK,
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }

    @Synchronized
    fun start() {
        check(!running) {
            "UDP remote evidence transport is already running"
        }

        if (!permissionGranted()) {
            throw SecurityException(
                "Local network permission is required for direct UDP capture transport on Android 17+",
            )
        }

        val datagramSocket =
            DatagramSocket()

        datagramSocket.reuseAddress = true
        datagramSocket.receiveBufferSize =
            config.receiveBufferBytes
        datagramSocket.soTimeout =
            config.receiveTimeoutMillis

        datagramSocket.bind(
            InetSocketAddress(
                config.bindPort,
            ),
        )

        socket = datagramSocket
        running = true

        decodeExecutor.purge()

        executor.execute {
            receiveLoop(
                datagramSocket,
            )
        }
    }

    @Synchronized
    fun stop() {
        running = false
        socket?.close()
        socket = null
        decodeExecutor.clearQueuedTasks()
        reassembler.clear()
        replayGuard.clearAll()
    }

    fun register(
        endpoint: Endpoint,
    ) {
        require(peers.size < config.maximumPeers || peers.containsKey(endpoint)) {
            "UDP peer endpoint capacity exhausted"
        }

        peers[endpoint] =
            endpoint.address
    }

    fun unregister(
        endpoint: Endpoint,
    ) {
        peers.remove(endpoint)
        reassembler.clearSource(
            endpoint.toString(),
        )
        replayGuard.clear(
            endpoint.toString(),
        )
    }

    fun registeredEndpoints(): List<Endpoint> =
        peers.keys
            .sortedWith(
                compareBy<Endpoint> {
                    it.host
                }.thenBy {
                    it.port
                },
            )

    fun sendSecure(
        endpoint: Endpoint,
        origin: DeviceId,
        destination: DeviceId,
        sequence: Long,
        timestampNanos: Long,
        plaintext: ByteArray,
    ): Boolean {
        check(running) {
            "UDP remote evidence transport is not running"
        }

        val target =
            peers[endpoint]
                ?: run {
                    register(endpoint)
                    peers[endpoint]!!
                }

        val securePacket =
            SecureRemoteEvidenceCodec.encode(
                origin = origin,
                destination = destination,
                sequence = sequence,
                timestampNanos = timestampNanos,
                plaintext = plaintext,
                key = securityKey,
            )

        require(
            securePacket.size <=
                config.maximumPacketBytes,
        )

        val datagrams =
            RemoteDatagramFragmentCodec.fragment(
                payload = securePacket,
                messageSequence = sequence,
            )

        require(
            datagrams.all {
                it.size <=
                    config.maximumDatagramBytes
            },
        ) {
            "Configured UDP datagram budget is below the transport fragment size"
        }

        return runCatching {
            val datagramSocket =
                socket
                    ?: error("UDP socket is not running")

            datagrams.forEach { bytes ->
                require(
                    bytes.size <=
                        config.maximumDatagramBytes,
                )

                datagramSocket.send(
                    DatagramPacket(
                        bytes,
                        bytes.size,
                        target,
                    ),
                )
                sentDatagrams.incrementAndGet()
            }

            true
        }.getOrDefault(false)
    }

    fun diagnostics(): Diagnostics =
        Diagnostics(
            running = running,
            bindPort =
                socket?.localPort
                    ?: config.bindPort,
            receivedDatagrams =
                receivedDatagrams.get(),
            droppedDatagrams =
                droppedDatagrams.get(),
            decodedPackets =
                decodedPackets.get(),
            authenticationFailures =
                authenticationFailures.get(),
            replayRejectedPackets =
                replayGuard.diagnostics()
                    .rejected,
            inFlightAssemblies =
                reassembler.inFlightAssemblyCount(),
            inFlightBytes =
                reassembler.inFlightBytes(),
            decodeQueueDepth =
                decodeExecutor.queuedTasks(),
            decodeOutstandingTasks =
                decodeExecutor.outstandingTasks(),
            permissionGranted =
                permissionGranted(),
        )

    override fun close() {
        stop()
        peers.clear()
        executor.shutdownNow()
        decodeExecutor.close()
    }

    private fun securePacketTimestampNanos(
        securePacket: ByteArray,
    ): Long {
        if (securePacket.size < 36) {
            return 0L
        }

        return runCatching {
            java.nio.ByteBuffer
                .wrap(securePacket)
                .order(java.nio.ByteOrder.BIG_ENDIAN)
                .getLong(20)
                .coerceAtLeast(0L)
        }.getOrDefault(0L)
    }

    private fun receiveLoop(
        datagramSocket: DatagramSocket,
    ) {
        val buffer =
            ByteArray(
                config.maximumDatagramBytes,
            )

        while (running) {
            try {
                val packet =
                    DatagramPacket(
                        buffer,
                        buffer.size,
                    )

                datagramSocket.receive(packet)
                receivedDatagrams.incrementAndGet()

                val bytes =
                    packet.data.copyOfRange(
                        packet.offset,
                        packet.offset +
                            packet.length,
                    )

                if (
                    packet.length >
                        config.maximumDatagramBytes
                ) {
                    droppedDatagrams.incrementAndGet()
                    continue
                }

                val source =
                    packet.address.hostAddress.orEmpty() +
                        ":" +
                        packet.port

                val securePacket =
                    runCatching {
                        reassembler.accept(
                            source = source,
                            datagram = bytes,
                            nowNanos = System.nanoTime(),
                        )
                    }.getOrElse {
                        droppedDatagrams.incrementAndGet()
                        null
                    }
                    ?: continue

                if (
                    securePacket.size >
                        config.maximumPacketBytes
                ) {
                    droppedDatagrams.incrementAndGet()
                    continue
                }

                val sourceAddress =
                    InetSocketAddress(
                        packet.address,
                        packet.port,
                    )

                val priorityTimestamp =
                    if (config.decodeFreshnessPriorityEnabled) {
                        securePacketTimestampNanos(
                            securePacket,
                        )
                    } else {
                        0L
                    }

                val admitted =
                    decodeExecutor.tryExecute(
                        timestampNanos = priorityTimestamp,
                    ) {
                        if (!running) {
                            return@tryExecute
                        }

                        if (
                            !replayGuard.accept(
                                source,
                                securePacket,
                            )
                        ) {
                            return@tryExecute
                        }

                        val decoded =
                            runCatching {
                                SecureRemoteEvidenceCodec.decode(
                                    packetBytes = securePacket,
                                    key = securityKey,
                                )
                            }.getOrElse {
                                authenticationFailures.incrementAndGet()
                                reassembler.clearSource(source)
                                return@execute
                            }

                        decodedPackets.incrementAndGet()

                        _received.tryEmit(
                            Received(
                                source = sourceAddress,
                                securePacket = securePacket,
                                decoded = decoded,
                            ),
                        )
                    }

                if (!admitted) {
                    droppedDatagrams.incrementAndGet()
                }
            } catch (_: java.net.SocketTimeoutException) {
                reassembler.expire(System.nanoTime())
            } catch (_: SocketException) {
                if (running) {
                    droppedDatagrams.incrementAndGet()
                }
            } catch (_: Throwable) {
                if (running) {
                    droppedDatagrams.incrementAndGet()
                }
            }

            reassembler.expire(
                System.nanoTime(),
            )
        }
    }
}
