# for making comparison reports, we can target the 6Mo files that have 2Mo
# prerequisites. fake('tcircevo/*b-ave.fif') will create list of 6Mo path
# strings that can be safely used by report generation function to also open
# the 2Mo files (without need to check for existence), e.g.,
# makeCompreportFromTcircevos( '2to6MoComp.html', fake('tcircevo/*b-ave.fif') )

#tcircevo/%b-ave.fif : tcircevo/%a-ave.fif
#	echo $@

# in similar fashion, we can touch a batch of empty targets in bash:
# for i in ../tone/*/epochs/*epo.fif; do touch epofif/`basename ${i}`; done;
# and then use these as empty targets to generate lists of prereq arguments
# ("$?") that can be passed to report generator

epofif/%b-epo.fif : tcircevo/%b-ave.fif tcircevo/%a-ave.fif ../tone/*/epochs/%b-epo.fif ../tone/*/epochs/%a-epo.fif
	echo $@

2MoReport : tcircevo/*a-ave.fif
	echo makeReportFromTcircevos 2MoTCircs.html $?

6MoReport : tcircevo/*b-ave.fif
	echo makeReportFromTcircevos 6MoTCircs.html $?

tcircevo/%-ave.fif : ../tone/*/epochs/%-epo.fif
	echo makeTcircevoFromEpo $@ $<

