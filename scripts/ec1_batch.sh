#!/usr/bin/env bash
# n=1606 eval_curation_v1 validation: Arm A x2 + relnature x2 (medpsy-4B), sequential.
# Run detached on noot-1 (nohup/setsid) so it survives SSH drop / laptop sleep.
cd ~/indra-belief-model || exit 1
run(){ MONO_VARIANT="$1" PYTHONPATH=src .venv/bin/python scripts/run_rasmachine_monolithic.py \
  --model remote-medpsy-4b --input /tmp/eval_curation_v1_statements.json \
  --output "/tmp/$2.jsonl" --workers 4 --no-resume --row-error-policy record --no-export \
  >"/tmp/$2.runlog" 2>&1; }
rm -f /tmp/ec1_BATCH_DONE
run disconfirm           ec1_arma_r1
run disconfirm           ec1_arma_r2
run disconfirm_relnature ec1_relnat_r1
run disconfirm_relnature ec1_relnat_r2
touch /tmp/ec1_BATCH_DONE
