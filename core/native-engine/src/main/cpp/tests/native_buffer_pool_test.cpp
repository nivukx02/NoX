#include <cassert>
#include <cstddef>

#include "memory/native_buffer_pool.h"

int main() {
    nox::native::NativeBufferPool pool(4, 256);

    assert(pool.capacity()==4u);
    assert(pool.block_size()==256u);
    assert(pool.resident_bytes()==1024u);
    assert(pool.active_blocks()==0u);
    assert(pool.free_blocks()==4u);

    const auto a=pool.acquire();
    const auto b=pool.acquire();
    const auto c=pool.acquire();
    const auto d=pool.acquire();
    const auto exhausted=pool.acquire();

    assert(a>=0 && b>=0 && c>=0 && d>=0);
    assert(exhausted<0);
    assert(pool.active_blocks()==4u);
    assert(pool.free_blocks()==0u);

    pool.release(static_cast<std::uint32_t>(b));
    pool.release(static_cast<std::uint32_t>(b));
    assert(pool.active_blocks()==3u);
    assert(pool.free_blocks()==1u);

    const auto reused=pool.acquire();
    assert(reused==b);
    assert(pool.active_blocks()==4u);


    nox::native::NativeBufferPool handlePool(2, 128);

    const auto firstHandle=handlePool.acquire_handle();
    assert(firstHandle.valid);
    assert(handlePool.release(firstHandle));

    const auto reusedHandle=handlePool.acquire_handle();
    assert(reusedHandle.valid);
    assert(reusedHandle.index==firstHandle.index);
    assert(reusedHandle.generation!=firstHandle.generation);

    assert(!handlePool.release(firstHandle));
    assert(handlePool.stale_release_count()==1u);
    assert(handlePool.active_blocks()==1u);

    assert(handlePool.release(reusedHandle));
    assert(handlePool.active_blocks()==0u);

    pool.release(static_cast<std::uint32_t>(a));
    pool.release(static_cast<std::uint32_t>(c));
    pool.release(static_cast<std::uint32_t>(d));
    pool.release(static_cast<std::uint32_t>(reused));

    assert(pool.active_blocks()==0u);
    assert(pool.free_blocks()==4u);
    return 0;
}
