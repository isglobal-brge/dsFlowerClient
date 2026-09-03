#include <R.h>
#include <Rinternals.h>
#include <R_ext/Connections.h>

/*
 * R's connection API is versioned and may change incompatibly.  Keep the ABI
 * pin even though R >= 4.6 classifies this header as experimental API.
 * R's socket timeout path can report zero after writing a prefix; the R caller
 * treats zero for a nonempty payload as indeterminate and closes the socket.
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
