# -*- coding: utf-8 -*-

# Coding decl above needed for rendering the emdash properly in the
# documentation.

"""
Module ``ioutils`` implements a number of helper classes and functions which
are useful when dealing with input, output, and bytestreams in a variety of
ways.
"""
import os
from io import (
    BytesIO,
    RawIOBase,
    DEFAULT_BUFFER_SIZE,
    SEEK_SET, SEEK_CUR, SEEK_END,
    UnsupportedOperation,
)
from abc import (
    ABCMeta,
    abstractmethod,
    abstractproperty,
)
from errno import EINVAL
from codecs import EncodedFile
from tempfile import TemporaryFile

try:
    text_type = unicode  # Python 2
    binary_type = str
except NameError:
    text_type = str      # Python 3
    binary_type = bytes

READ_CHUNK_SIZE = 21333
"""
Number of bytes to read at a time. The value is ~ 1/3rd of 64k which means that
the value will easily fit in the L2 cache of most processors even if every
codepoint in a string is three bytes long which makes it a nice fast default
value.
"""


class SpooledIOBase(object):
    """
    The SpooledTempoaryFile class doesn't support a number of attributes and
    methods that a StringIO instance does. This brings the api as close to
    compatible as possible with StringIO so that it may be used as a near
    drop-in replacement to save memory.

    Another issue with SpooledTemporaryFile is that the spooled file is always
    a cStringIO rather than a StringIO which causes issues with some of our
    tools.
    """
    __metaclass__ = ABCMeta

    def __init__(self, max_size=5000000, dir=None):
        self._max_size = max_size
        self._dir = dir

    @abstractmethod
    def read(self, n=-1):
        """Read n characters from the buffer"""

    @abstractmethod
    def write(self, s):
        """Write into the buffer"""

    @abstractmethod
    def seek(self, pos, mode=0):
        """Seek to a specific point in a file"""

    @abstractmethod
    def readline(self, length=None):
        """Returns the next available line"""

    @abstractmethod
    def readlines(self, sizehint=0):
        """Returns a list of all lines from the current position forward"""

    @abstractmethod
    def rollover(self):
        """Roll file-like-object over into a real temporary file"""

    @abstractmethod
    def tell(self):
        """Return the current position"""

    @abstractproperty
    def buffer(self):
        """Should return a flo instance"""

    @abstractproperty
    def _rolled(self):
        """Returns whether the file has been rolled to a real file or not"""

    @abstractproperty
    def len(self):
        """Returns the length of the data"""

    def _get_softspace(self):
        return self.buffer.softspace

    def _set_softspace(self, val):
        self.buffer.softspace = val

    softspace = property(_get_softspace, _set_softspace)

    @property
    def _file(self):
        return self.buffer

    def close(self):
        return self.buffer.close()

    def flush(self):
        return self.buffer.flush()

    def isatty(self):
        return self.buffer.isatty()

    def next(self):
        line = self.readline()
        if not line:
            pos = self.buffer.tell()
            self.buffer.seek(0, os.SEEK_END)
            if pos == self.buffer.tell():
                raise StopIteration
            else:
                self.buffer.seek(pos)
        return line

    @property
    def closed(self):
        return self.buffer.closed

    @property
    def pos(self):
        return self.tell()

    @property
    def buf(self):
        return self.getvalue()

    def fileno(self):
        self.rollover()
        return self.buffer.fileno()

    def truncate(self, size=None):
        """
        Custom version of truncate that takes either no arguments (like the
        real SpooledTemporaryFile) or a single argument that truncates the
        value to a certain index location.
        """
        if size is None:
            return self.buffer.truncate()

        if size < 0:
            raise IOError(EINVAL, "Negative size not allowed")

        # Emulate truncation to a particular location
        pos = self.tell()
        self.seek(size)
        self.buffer.truncate()
        if pos < size:
            self.seek(pos)

    def getvalue(self):
        """Return the entire files contents"""
        pos = self.tell()
        self.seek(0)
        val = self.read()
        self.seek(pos)
        return val

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return True

    __next__ = next

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.close()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.getvalue() == other.getvalue()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return True

    __nonzero__ = __bool__


class SpooledBytesIO(SpooledIOBase):
    """
    SpooledBytesIO is a spooled file-like-object that only accepts bytes. On
    Python 2.x this means the 'str' type; on Python 3.x this means the 'bytes'
    type. Bytes are written in and retrieved exactly as given, but it will
    raise TypeErrors if something other than bytes are written.

    Example::

        >>> from boltons import ioutils
        >>> with ioutils.SpooledBytesIO() as f:
        ...     f.write(b"Happy IO")
        ...     _ = f.seek(0)
        ...     isinstance(f.getvalue(), ioutils.binary_type)
        True
    """

    def read(self, n=-1):
        return self.buffer.read(n)

    def write(self, s):
        if not isinstance(s, binary_type):
            raise TypeError("{0} expected, got {1}".format(
                binary_type.__name__,
                type(s).__name__
            ))

        if self.tell() + len(s) >= self._max_size:
            self.rollover()
        self.buffer.write(s)

    def seek(self, pos, mode=0):
        return self.buffer.seek(pos, mode)

    def readline(self, length=None):
        if length:
            return self.buffer.readline(length)
        else:
            return self.buffer.readline()

    def readlines(self, sizehint=0):
        return self.buffer.readlines(sizehint)

    def rollover(self):
        """Roll the StringIO over to a TempFile"""
        if not self._rolled:
            tmp = TemporaryFile(dir=self._dir)
            pos = self.buffer.tell()
            tmp.write(self.buffer.getvalue())
            tmp.seek(pos)
            self.buffer.close()
            self._buffer = tmp

    @property
    def _rolled(self):
        return not isinstance(self.buffer, BytesIO)

    @property
    def buffer(self):
        try:
            return self._buffer
        except AttributeError:
            self._buffer = BytesIO()
        return self._buffer

    @property
    def len(self):
        """Determine the length of the file"""
        pos = self.tell()
        if self._rolled:
            self.seek(0)
            val = os.fstat(self.fileno()).st_size
        else:
            self.seek(0, os.SEEK_END)
            val = self.tell()
        self.seek(pos)
        return val

    def tell(self):
        return self.buffer.tell()


class SpooledStringIO(SpooledIOBase):
    """
    SpooledStringIO is a spooled file-like-object that only accepts unicode
    values. On Python 2.x this means the 'unicode' type and on Python 3.x this
    means the 'str' type. Values are accepted as unicode and then coerced into
    utf-8 encoded bytes for storage. On retrieval, the values are returned as
    unicode.

    Example::

        >>> from boltons import ioutils
        >>> with ioutils.SpooledStringIO() as f:
        ...     f.write(u"\u2014 Hey, an emdash!")
        ...     _ = f.seek(0)
        ...     isinstance(f.read(), ioutils.text_type)
        True

    """
    def __init__(self, *args, **kwargs):
        self._tell = 0
        super(SpooledStringIO, self).__init__(*args, **kwargs)

    def read(self, n=-1):
        ret = self.buffer.reader.read(n, n)
        self._tell = self.tell() + len(ret)
        return ret

    def write(self, s):
        if not isinstance(s, text_type):
            raise TypeError("{0} expected, got {1}".format(
                text_type.__name__,
                type(s).__name__
            ))
        current_pos = self.tell()
        if self.buffer.tell() + len(s.encode('utf-8')) >= self._max_size:
            self.rollover()
        self.buffer.write(s.encode('utf-8'))
        self._tell = current_pos + len(s)

    def _traverse_codepoints(self, current_position, n):
        """Traverse from current position to the right n codepoints"""
        dest = current_position + n
        while True:
            if current_position == dest:
                # By chance we've landed on the right position, break
                break

            # If the read would take us past the intended position then
            # seek only enough to cover the offset
            if current_position + READ_CHUNK_SIZE > dest:
                self.read(dest - current_position)
                break
            else:
                ret = self.read(READ_CHUNK_SIZE)

            # Increment our current position
            current_position += READ_CHUNK_SIZE

            # If we kept reading but there was nothing here, break
            # as we are at the end of the file
            if not ret:
                break

        return dest

    def seek(self, pos, mode=0):
        """Traverse from offset to the specified codepoint"""
        # Seek to position from the start of the file
        if mode == os.SEEK_SET:
            self.buffer.seek(0)
            self._traverse_codepoints(0, pos)
            self._tell = pos
        # Seek to new position relative to current position
        elif mode == os.SEEK_CUR:
            start_pos = self.tell()
            self._traverse_codepoints(self.tell(), pos)
            self._tell = start_pos + pos
        elif mode == os.SEEK_END:
            self.buffer.seek(0)
            dest_position = self.len - pos
            self._traverse_codepoints(0, dest_position)
            self._tell = dest_position
        else:
            raise ValueError(
                "Invalid whence ({0}, should be 0, 1, or 2)".format(mode)
            )
        return self.tell()

    def readline(self, length=None):
        ret = self.buffer.readline(length).decode('utf-8')
        self._tell = self.tell() + len(ret)
        return ret

    def readlines(self, sizehint=0):
        ret = [x.decode('utf-8') for x in self.buffer.readlines(sizehint)]
        self._tell = self.tell() + sum((len(x) for x in ret))
        return ret

    @property
    def buffer(self):
        try:
            return self._buffer
        except AttributeError:
            self._buffer = EncodedFile(BytesIO(), data_encoding='utf-8')
        return self._buffer

    @property
    def _rolled(self):
        return not isinstance(self.buffer.stream, BytesIO)

    def rollover(self):
        """Roll the StringIO over to a TempFile"""
        if not self._rolled:
            tmp = EncodedFile(TemporaryFile(dir=self._dir),
                              data_encoding='utf-8')
            pos = self.buffer.tell()
            tmp.write(self.buffer.getvalue())
            tmp.seek(pos)
            self.buffer.close()
            self._buffer = tmp

    def tell(self):
        """Return the codepoint position"""
        return self._tell

    @property
    def len(self):
        """Determine the number of codepoints in the file"""
        pos = self.buffer.tell()
        self.buffer.seek(0)
        total = 0
        while True:
            ret = self.read(READ_CHUNK_SIZE)
            if not ret:
                break
            total += len(ret)
        self.buffer.seek(pos)
        return total


def is_text_fileobj(fileobj):
    if getattr(fileobj, 'encoding', False):
        # codecs.open and io.TextIOBase
        return True
    if getattr(fileobj, 'getvalue', False):
        # StringIO.StringIO / cStringIO.StringIO / io.StringIO
        try:
            if isinstance(fileobj.getvalue(), type(u'')):
                return True
        except Exception:
            pass
    return False


class MultiFileReader(object):
    """Takes a list of open files or file-like objects and provides an
    interface to read from them all contiguously. Like
    :func:`itertools.chain()`, but for reading files.

       >>> mfr = MultiFileReader(BytesIO(b'ab'), BytesIO(b'cd'), BytesIO(b'e'))
       >>> mfr.read(3).decode('ascii')
       u'abc'
       >>> mfr.read(3).decode('ascii')
       u'de'

    The constructor takes as many fileobjs as you hand it, and will
    raise a TypeError on non-file-like objects. A ValueError is raised
    when file-like objects are a mix of bytes- and text-handling
    objects (for instance, BytesIO and StringIO).
    """

    def __init__(self, *fileobjs):
        if not all([callable(getattr(f, 'read', None)) and
                    callable(getattr(f, 'seek', None)) for f in fileobjs]):
            raise TypeError('MultiFileReader expected file-like objects'
                            ' with .read() and .seek()')
        if all([is_text_fileobj(f) for f in fileobjs]):
            # codecs.open and io.TextIOBase
            self._joiner = u''
        elif any([is_text_fileobj(f) for f in fileobjs]):
            raise ValueError('All arguments to MultiFileReader must handle'
                             ' bytes OR text, not a mix')
        else:
            # open/file and io.BytesIO
            self._joiner = b''
        self._fileobjs = fileobjs
        self._index = 0

    def read(self, amt=None):
        """Read up to the specified *amt*, seamlessly bridging across
        files. Returns the appropriate type of string (bytes or text)
        for the input, and returns an empty string when the files are
        exhausted.
        """
        if not amt:
            return self._joiner.join(f.read() for f in self._fileobjs)
        parts = []
        while amt > 0 and self._index < len(self._fileobjs):
            parts.append(self._fileobjs[self._index].read(amt))
            got = len(parts[-1])
            if got < amt:
                self._index += 1
            amt -= got
        return self._joiner.join(parts)

    def seek(self, offset, whence=os.SEEK_SET):
        """Enables setting position of the file cursor to a given
        *offset*. Currently only supports ``offset=0``.
        """
        if whence != os.SEEK_SET:
            raise NotImplementedError(
                'MultiFileReader.seek() only supports os.SEEK_SET')
        if offset != 0:
            raise NotImplementedError(
                'MultiFileReader only supports seeking to start at this time')
        for f in self._fileobjs:
            f.seek(0)


class SeekableStreamReader(RawIOBase):
    def __init__(self, stream, buffer_size=0, chunk_size=DEFAULT_BUFFER_SIZE):
        self.stream = stream
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.buffer = bytearray(buffer_size)
        self.spos = 0
        self.vpos = 0

    def readable(self):
        return True

    def writeable(self):
        return False

    def seekable(self):
        # we can only guarantee generic seekability with unlimited buffer
        return not self.buffer_size

    def getbuffer(self):
        return memoryview(self.buffer)

    def tell(self):
        return self.vpos

    def _read_until(self, offset):
        self.vpos = self.spos
        buf = bytearray(self.chunk_size)

        if offset < 0:
            while self.readinto(buf) != 0:
                pass

        else:
            buf = memoryview(buf)
            while True:
                rest = offset - self.spos
                if not rest or self.readinto(buf[:rest]) == 0:
                    break

    def seek(self, offset, whence=SEEK_SET):
        # from end of file
        if whence == SEEK_END:
            # why would anyone even try that?
            if offset >= 0:
                raise UnsupportedOperation(
                    f"can't do non-negative end-relative seeks")

            # in a stream we only know the end, when we see it, thus the
            # buffer needs to be big enough
            if self.buffer_size and -offset > self.buffer_size:
                raise UnsupportedOperation(
                    f"buffer too small to seek to end-relative {offset}")

            # read until we are at the end then jump back in buffer
            self._read_until(offset)
            self.vpos = self.spos + offset
            return self.vpos

        # turn relative into absolute
        if whence == SEEK_CUR:
            offset = self.vpos + offset

        # safety check
        if offset < 0:
            raise ValueError(f"negative seek position {offset}")

        # jump forward in stream
        elif offset > self.spos:
            self._read_until(offset)

        # going backwards more than the buffer allows?
        elif offset < self.spos - len(self.buffer):
            raise UnsupportedOperation(
                f"buffer too small to seek to {offset}")

        else:
            self.vpos = offset

        return self.vpos

    def readinto(self, b):
        # read from buffer
        if self.vpos < self.spos:
            l = self.spos - self.vpos
            b[:l] = self.buffer[-l:]
            self.vpos = self.spos
            return l

        # actually read from stream
        else:
            l = self.stream.readinto(b)
            if l:
                # unlimited buffer: just extend
                if not self.buffer_size:
                    self.buffer.extend(b[:l])

                # bigger than buffer: replace
                elif l >= self.buffer_size:
                    self.buffer[:] = b[l-self.buffer_size:l]

                # smaller than buffer: copy end to front, then extend
                else:
                    keep = self.buffer_size - l
                    self.buffer[:keep] = self.buffer[-keep:]
                    self.buffer[keep:] = b[:l]

                self.vpos = self.spos = self.spos + l

            return l
