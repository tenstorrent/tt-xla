def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--folder",
        action="store",
        default="./collected_irs",
        help="Folder path containing IR files to process (default: ./collected_irs)"
    )
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Only compile ops without execution (default: False)"
    )
