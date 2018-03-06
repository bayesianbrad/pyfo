We do need the accumaltative value of the logpdf throughout the program.
    But we need to store the discrete logpdf separately from the continuious
log_pdf.
    So we can see x['log_pdf'] = 0 on the outside of the discrete loop and when that has
finished we then can add the log_pdf genearatded from here, to the log_pdf geenrated from the
first update of the continuous prameters. Check with nishimura.

    Things tried:
    Setting x['log_pdf'] = 0 at the end of the run, does not help.
