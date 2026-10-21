"""Raw -> clean parsers: one module per customs publication.

Each parser turns the raw retrievals of one customs book, read from
``data/customs/<book>/raw/``, into the standardized long CSV described in ``_common``, written
to ``data/customs/<book>/clean/``. Parsers never touch the network.

Raw files
    The raw files are archives in the format written by the project's download scripts,
    which are not published. For most books an archive is the publisher's responses, kept
    verbatim, wrapped in a small JSON envelope that records the retrieval (the codes, flows
    and periods requested; the publication frontier where the publisher states one) and the
    reference tables the responses need (partner names, code descriptions). Taiwan and China
    offer no usable API: their raw files are manual exports from the publishers' query
    portals (``.xls`` and ``.csv``), saved as downloaded. Each module's docstring documents
    the exact layout it expects, so the files can be reproduced with any downloader;
    obtaining them is up to the user, under each publisher's terms (docs/data.md lists the
    publications). Every raw file name carries its retrieval date as ``pulled<YYYYMMDD>``.

Parser contract
    Every parser module exposes

        BOOK                                     the book directory under data/customs/ it
                                                 builds by default (None for ``comtrade``,
                                                 which serves the books in its ``BOOKS``)
        build(raw_dir, fx_rates, vintage=None)   -> list[CleanBook]

    ``build`` reads the raw retrievals in ``raw_dir``:

    * books that receive trailing-period refreshes (UK, Japan, Korea, India, Indonesia,
      Thailand, Taiwan, China) merge every retrieval, the newest retrieval winning per record;
      each module documents its merge key. ``vintage="YYYYMMDD"`` ignores the retrievals made
      after that date, rebuilding the book as it stood then.
    * the other books re-read their whole series at every retrieval, so ``build`` reads the
      newest retrieval only; ``vintage="YYYYMMDD"`` pins the retrieval stamped that day.

    ``fx_rates`` is the table returned by ``_common.load_fx_rates``; books published in US
    dollars ignore it. ``pulled_date`` on every row is a retrieval date taken from the raw
    files, never the build date.

Command line
    python -m ai_chip_imports.sources <parser> [--book BOOK] [--raw-dir DIR] [--clean-dir DIR]
                                             [--fx CSV] [--vintage YYYYMMDD]

    ``--book`` selects the book directory. It is required for ``comtrade``, which serves the
    books published only through UN Comtrade (Saudi Arabia, Australia, Canada) and the
    Comtrade companions of the Malaysia and Mexico books.
"""

PARSERS = (
    "us", "taiwan", "china", "eu", "uk", "japan", "korea", "hk", "singapore", "malaysia",
    "thailand", "indonesia", "india", "mexico", "comtrade",
)
