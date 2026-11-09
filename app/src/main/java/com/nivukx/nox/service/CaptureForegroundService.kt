package com.nivukx.nox.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder

class CaptureForegroundService : Service() {
    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                "nox_capture",
                "NoX Capture",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
        startForeground(
            1001,
            Notification.Builder(this, "nox_capture")
                .setContentTitle("NoX distributed capture")
                .setContentText("Synchronized multi-camera motion capture is active")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setOngoing(true)
                .build(),
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int =
        START_NOT_STICKY

    override fun onBind(intent: Intent?): IBinder? = null
}
