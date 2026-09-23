from pathlib import Path
import ctypes
from ctypes.util import find_library
import sys

import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg


class Rectangle(ctypes.Structure):
    _fields_ = [(field, ctypes.c_double) for field in ("x", "y", "width", "height")]


class GError(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint), ("code", ctypes.c_int), ("message", ctypes.c_char_p)]


source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
handle = Rsvg.Handle.new_from_file(str(source))
valid, width, height = handle.get_intrinsic_size_in_pixels()
if not valid or width <= 0 or height <= 0:
    raise ValueError("The schematic must declare physical dimensions")
target.parent.mkdir(parents=True, exist_ok=True)
rsvg = ctypes.CDLL(find_library("rsvg-2"))
cairo = ctypes.CDLL(find_library("cairo"))
gobject = ctypes.CDLL(find_library("gobject-2.0"))
glib = ctypes.CDLL(find_library("glib-2.0"))
error_pointer = ctypes.POINTER(GError)
rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(error_pointer)]
rsvg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
rsvg.rsvg_handle_render_document.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Rectangle), ctypes.POINTER(error_pointer)]
rsvg.rsvg_handle_render_document.restype = ctypes.c_int
cairo.cairo_pdf_surface_create.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
cairo.cairo_pdf_surface_create.restype = ctypes.c_void_p
cairo.cairo_create.argtypes = [ctypes.c_void_p]
cairo.cairo_create.restype = ctypes.c_void_p
for function in ("cairo_destroy", "cairo_surface_finish", "cairo_surface_destroy"):
    getattr(cairo, function).argtypes = [ctypes.c_void_p]
    getattr(cairo, function).restype = None
cairo.cairo_surface_status.argtypes = [ctypes.c_void_p]
cairo.cairo_surface_status.restype = ctypes.c_int
gobject.g_object_unref.argtypes = [ctypes.c_void_p]
gobject.g_object_unref.restype = None
glib.g_error_free.argtypes = [error_pointer]
glib.g_error_free.restype = None
error = error_pointer()
native_handle = rsvg.rsvg_handle_new_from_file(str(source).encode(), ctypes.byref(error))
if not native_handle:
    message = error.contents.message.decode() if error else "SVG loading failed"
    if error:
        glib.g_error_free(error)
    raise RuntimeError(message)
surface = cairo.cairo_pdf_surface_create(str(target).encode(), width * 0.75, height * 0.75)
context = cairo.cairo_create(surface)
try:
    viewport = Rectangle(0, 0, width * 0.75, height * 0.75)
    if not rsvg.rsvg_handle_render_document(native_handle, context, ctypes.byref(viewport), ctypes.byref(error)):
        message = error.contents.message.decode() if error else "SVG rendering failed"
        if error:
            glib.g_error_free(error)
        raise RuntimeError(message)
    cairo.cairo_surface_finish(surface)
    if cairo.cairo_surface_status(surface) != 0:
        raise RuntimeError("Cairo PDF surface reported an error")
finally:
    cairo.cairo_destroy(context)
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(native_handle)
print(f"Rendered vector schematic: {target.name}")
