// The prebuilt onnxruntime aarch64 static lib is compiled with GCC outline
// atomics and references __aarch64_cas8_sync. That helper lives only in
// libgcc.a (hidden symbol, not in libgcc_s.so.1), and the version of
// libgcc.a in the manylinux_2_28 cross-toolchain (GCC 8) predates its
// introduction (GCC 9.3.1). Define it here instead.
//
// LLVM-generated Rust atomics never call __aarch64_cas8_sync, so there
// is no recursion risk.

use std::sync::atomic::{AtomicU64, Ordering};

#[no_mangle]
pub unsafe extern "C" fn __aarch64_cas8_sync(
    expected: u64,
    desired: u64,
    ptr: *mut u64,
) -> u64 {
    let atom = &*(ptr as *const AtomicU64);
    match atom.compare_exchange(expected, desired, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(v) | Err(v) => v,
    }
}
