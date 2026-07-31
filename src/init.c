#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

extern SEXP dsf_socket_write(SEXP, SEXP);

static const R_CallMethodDef call_methods[] = {
    {"dsf_socket_write", (DL_FUNC) &dsf_socket_write, 2},
    {NULL, NULL, 0}
};

void attribute_visible R_init_dsFlowerClient(DllInfo *dll) {
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
