bash ./C1L1P-B/train_C1L1P-B_loocv.sh
bash ./C1L1P-C/train_C1L1P-C_loocv.sh
bash ./C2L1P-B/train_C2L1P-B_loocv.sh
bash ./C2L1P-C/train_C2L1P-C_loocv.sh
echo "writing, nowriting is complete. Vj needs to run validations now." | mail -aFrom:rtx3@ivPCL.ece.unm.edu -c venkatesh369@unm.edu -s "MMACTION2 update" venkatesh.jatla@gmail.com sssravani4@gmail.com
