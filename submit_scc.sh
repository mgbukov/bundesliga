#!bin/bash

for i in 1 2 # 1: cont actions; 2: bang-bang
do

	for j in $(seq 1 1 100) # of random realisations
	do

		for k in $(seq 1 1 11) # of total ramp times
		do

			echo "#!/bin/bash -login" >> submission.sh
	        echo "#$ -P f-dmrg" >> submission.sh #evolve is name of job
	        echo "#$ -N RL_${i}_${j}_${k}" >> submission.sh #evolve is name of job
	        echo "#$ -l h_rt=2:59:00">> submission.sh #336
	        #echo "#$ -pe omp 4">> submission.sh #request more processors
	        echo "#$ -m ae" >> submission.sh #a (abort) e(exit)
	        #echo "#$-l mem_per_core=4G" >> submission.sh # more memory
	        #echo "#$ -M mbukov@bu.edu" >> submission.sh
	        echo "#$ -m n" >> submission.sh # disable emails
	        echo "source activate BL" >> submission.sh # requires conda env called 'ED' with quspin
	        echo "python scc.py $i $j $k" >> submission.sh
	        
	        qsub submission.sh
	        rm submission.sh
	        
	        sleep 1.0 #wait for half a second
	    
	    done

	done


done