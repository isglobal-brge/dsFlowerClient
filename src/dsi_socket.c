#include <R.h>
#include <Rinternals.h>
#include <R_ext/Connections.h>

/*
 * R has no public API exposing a non-blocking connection's partial-write count.
 * Lossless relay acknowledgements require that count, so fail at compile time
 * on an ABI change instead of silently assuming that a write was complete.
 */
#if R_CONNECTIONS_VERSION != 1
#error "Unsupported R connection API version"
#endif

SEXP dsf_socket_write(SEXP connection, SEXP payload) {
    if (TYPEOF(payload) != RAWSXP) {
        Rf_error("Tunnel socket payload must be raw bytes.");
    }

    R_xlen_t length = XLENGTH(payload);
    if (length == 0) {
        return Rf_ScalarReal(0);
    }

    Rconnection con = R_GetConnection(connection);
    size_t written = R_WriteConnection(
        con, (void *) RAW(payload), (size_t) length
    );
    return Rf_ScalarReal((double) written);
}
