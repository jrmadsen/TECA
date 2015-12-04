#ifndef teca_py_array_h
#define teca_py_array_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <cstdlib>

#include "teca_common.h"
#include "teca_variant_array.h"

namespace teca_py_array
{
/// cpp_tt -- traits class for working with PyArrayObject's
/**
cpp_tt::type -- get the C++ type given a numpy enum.

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <int numpy_code> struct cpp_tt
{};

#define teca_py_array_cpp_tt_declare(CODE, CPP_T)   \
template <> struct cpp_tt<CODE>                     \
{                                                   \
    typedef CPP_T type;                             \
};
teca_py_array_cpp_tt_declare(NPY_BYTE, char)
teca_py_array_cpp_tt_declare(NPY_INT32, int)
teca_py_array_cpp_tt_declare(NPY_INT64, long long)
teca_py_array_cpp_tt_declare(NPY_UBYTE, unsigned char)
teca_py_array_cpp_tt_declare(NPY_UINT32, unsigned int)
teca_py_array_cpp_tt_declare(NPY_UINT64, unsigned long long)
teca_py_array_cpp_tt_declare(NPY_FLOAT, float)
teca_py_array_cpp_tt_declare(NPY_DOUBLE, double)


/// numpy_tt - traits class for working with PyArrayObject's
/**
::code - get the numpy type enum given a C++ type.
::is_type - return true if the PyArrayObject has the given type

CODE -- numpy type enumeration
CPP_T -- corresponding C++ type
*/
template <typename cpp_t> struct numpy_tt
{};

#define teca_py_array_numpy_tt_declare(CODE, CPP_T) \
template <> struct numpy_tt<CPP_T>                  \
{                                                   \
    enum { code = CODE };                           \
    static bool is_type(PyArrayObject *arr)         \
    { return PyArray_TYPE(arr) == CODE; }           \
};
teca_py_array_numpy_tt_declare(NPY_BYTE, char)
teca_py_array_numpy_tt_declare(NPY_INT16, short)
teca_py_array_numpy_tt_declare(NPY_INT32, int)
teca_py_array_numpy_tt_declare(NPY_LONG, long)
teca_py_array_numpy_tt_declare(NPY_INT64, long long)
teca_py_array_numpy_tt_declare(NPY_UBYTE, unsigned char)
teca_py_array_numpy_tt_declare(NPY_UINT16, unsigned short)
teca_py_array_numpy_tt_declare(NPY_UINT32, unsigned int)
teca_py_array_numpy_tt_declare(NPY_ULONG, unsigned long)
teca_py_array_numpy_tt_declare(NPY_UINT64, unsigned long long)
teca_py_array_numpy_tt_declare(NPY_FLOAT, float)
teca_py_array_numpy_tt_declare(NPY_DOUBLE, double)


// CPP_T - array type to match
// OBJ - PyArrayObject* instance
// CODE - code to execute on match
#define TECA_PY_ARRAY_DISPATCH_CASE(CPP_T, OBJ, CODE)   \
    if (teca_py_array::numpy_tt<CPP_T>::is_type(OBJ))   \
    {                                                   \
        using AT = CPP_T;                               \
        CODE                                            \
    }

#define TECA_PY_ARRAY_DISPATCH(OBJ, CODE)                       \
    TECA_PY_ARRAY_DISPATCH_CASE(float, OBJ, CODE)               \
    TECA_PY_ARRAY_DISPATCH_CASE(double, OBJ, CODE)              \
    TECA_PY_ARRAY_DISPATCH_CASE(int, OBJ, CODE)                 \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned int, OBJ, CODE)        \
    TECA_PY_ARRAY_DISPATCH_CASE(long, OBJ, CODE)                \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned long, OBJ, CODE)       \
    TECA_PY_ARRAY_DISPATCH_CASE(long long, OBJ, CODE)           \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned long long, OBJ, CODE)  \
    TECA_PY_ARRAY_DISPATCH_CASE(char, OBJ, CODE)                \
    TECA_PY_ARRAY_DISPATCH_CASE(unsigned char, OBJ, CODE)

// ****************************************************************************
bool append(teca_variant_array *varr, PyObject *obj)
{
    // not an array
    if (!PyArray_Check(obj))
        return false;

    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

    // nothing to do.
    size_t n_elem = PyArray_SIZE(arr);
    if (!n_elem)
        return true;

    // append
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);

        TECA_PY_ARRAY_DISPATCH(arr,
            PyObject *it = PyArray_IterNew(obj);
            for (size_t i = 0; i < n_elem; ++i)
            {
                varrt->append(*static_cast<AT*>(PyArray_ITER_DATA(it)));
                PyArray_ITER_NEXT(it);
            }
            Py_DECREF(it);
            return true;
            )
        )

    // unknown type
    return false;
}

// ****************************************************************************
bool copy(teca_variant_array *varr, PyObject *obj)
{
    // not an array
    if (!PyArray_Check(obj))
        return false;

    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

    // nothing to do.
    size_t n_elem = PyArray_SIZE(arr);
    if (!n_elem)
        return true;

    // copy
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,

        TT *varrt = static_cast<TT*>(varr);
        varrt->resize(n_elem);

        TECA_PY_ARRAY_DISPATCH(arr,
            PyObject *it = PyArray_IterNew(obj);
            for (size_t i = 0; i < n_elem; ++i)
            {
                varrt->set(i, *static_cast<AT*>(PyArray_ITER_DATA(it)));
                PyArray_ITER_NEXT(it);
            }
            Py_DECREF(it);
            return true;
            )
        )

    // unknown type
    return false;
}

// ****************************************************************************
p_teca_variant_array new_variant_array(PyObject *obj)
{
    // not an array
    if (!PyArray_Check(obj))
        return nullptr;

    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj);

    // allocate and copy
    TECA_PY_ARRAY_DISPATCH(arr,
        size_t n_elem = PyArray_SIZE(arr);

        p_teca_variant_array_impl<AT> varr
             = teca_variant_array_impl<AT>::New(n_elem);

        PyObject *it = PyArray_IterNew(obj);
        for (size_t i = 0; i < n_elem; ++i)
        {
            varr->set(i, *static_cast<AT*>(PyArray_ITER_DATA(it)));
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(it);

        return varr;
        )

    // unknown type
    return nullptr;
}

// ****************************************************************************
template <typename NT>
PyArrayObject *new_object(teca_variant_array_impl<NT> *varrt)
{
    // allocate a buffer
    npy_intp n_elem = varrt->size();
    size_t n_bytes = n_elem*sizeof(NT);
    NT *mem = static_cast<NT*>(malloc(n_bytes));
    if (!mem)
    {
        PyErr_Format(PyExc_RuntimeError,
            "failed to allocate %lu bytes", n_bytes);
        return nullptr;
    }

    // copy the data
    memcpy(mem, varrt->get(), n_bytes);

    // put the buffer in to a new numpy object
    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNewFromData(1, &n_elem, numpy_tt<NT>::code, mem));
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    return arr;
}

// ****************************************************************************
PyArrayObject *new_object(teca_variant_array *varr)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl, varr,
        TT *varrt = static_cast<TT*>(varr);
        return new_object(varrt);
        )
    return nullptr;
}
};

#endif
