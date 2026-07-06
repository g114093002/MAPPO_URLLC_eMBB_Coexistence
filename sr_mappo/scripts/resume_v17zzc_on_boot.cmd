@echo off
setlocal

cd /d D:\URLLC_eMBB_Coexisting

if not exist D:\URLLC_eMBB_Coexisting\logs mkdir D:\URLLC_eMBB_Coexisting\logs

powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*v17zzc_block_power_first_v1*' }) { exit 1 } else { exit 0 }"
if %errorlevel%==1 (
  echo [%date% %time%] Existing v17zzc training process detected; skip auto-resume >> D:\URLLC_eMBB_Coexisting\logs\resume_v17zzc_on_boot.log
  endlocal
  exit /b 0
)

echo [%date% %time%] Starting auto-resume for v17zzc training >> D:\URLLC_eMBB_Coexisting\logs\resume_v17zzc_on_boot.log

C:\python36\python.exe -m sr_mappo.train_clean_mappo ^
  --experiment v17zzc_block_power_first_v1 ^
  --resume-from "D:\URLLC_eMBB_Coexisting\checkpoints\clean_mappo\sr_mappo_tp_full_mappo_v17zzc_block_power_first_v1_clean_interrupted.pt" ^
  --iterations 10000 ^
  --rollout-horizon 256 ^
  --rollout-horizon-env-steps 108 ^
  --num-rollout-envs 4 ^
  --parallel-rollout-workers 2 ^
  --eval-every 10 ^
  --eval-episodes 2 ^
  --seed 42 >> D:\URLLC_eMBB_Coexisting\logs\resume_v17zzc_on_boot.log 2>&1

echo [%date% %time%] Auto-resume command exited with code %errorlevel% >> D:\URLLC_eMBB_Coexisting\logs\resume_v17zzc_on_boot.log

endlocal
