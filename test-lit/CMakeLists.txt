configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(TAFFO_TEST_DEPENDS
    Taffo)

add_lit_testsuite(taffo-check-lit "Running the regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${TAFFO_TEST_DEPENDS})
