import subprocess
import sys

commands = [
    "python base_SC.py --dataset Quake_10x_Limb_Muscle",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Quake_10x_Limb_Muscle",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Quake_10x_Limb_Muscle --epsilon 10",
    "python base_SC.py --dataset yan",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset yan",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset yan --epsilon 10",
    "python base_SC.py --dataset Bach",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Bach",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Bach --epsilon 10",
    "python base_SC.py --dataset goolam",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset goolam",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset goolam --epsilon 10",
    "python base_SC.py --dataset Melanoma_5K",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Melanoma_5K",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Melanoma_5K --epsilon 10",
    "python base_SC.py --dataset Muraro",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Muraro",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Muraro --epsilon 10",
    "python base_SC.py --dataset petropoulos",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset petropoulos",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset petropoulos --epsilon 10",
    "python base_SC.py --dataset Pollen",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Pollen",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Pollen --epsilon 10",
    "python base_SC.py --dataset Quake_10x_Bladder",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Quake_10x_Bladder",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Quake_10x_Bladder --epsilon 10",
    "python base_SC.py --dataset Quake_Smart-seq2_Diaphragm",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Quake_Smart-seq2_Diaphragm",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Quake_Smart-seq2_Diaphragm --epsilon 10",
    "python base_SC.py --dataset Shekhar",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Shekhar",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Shekhar --epsilon 10",
    "python base_SC.py --dataset Tosches",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Tosches",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Tosches --epsilon 10",
    "python base_SC.py --dataset worm_neuron_cell",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset worm_neuron_cell",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset worm_neuron_cell --epsilon 10",
    "python base_SC.py --dataset Young",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset Young",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset Young --epsilon 10",

    
    ##单独跑
    "python base_SC.py --dataset hrvatin",
    "python DL_SC.py --com-time 100 --client 5 --epochs 10 --dataset hrvatin",
    "python PFDL_SC.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset hrvatin --epsilon 10",

]

for idx, cmd in enumerate(commands, 1):
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(commands)}] Running command:")
    print(cmd)
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        shell=True
    )

### 如果你希望某个命令失败就直接停止
if result.returncode != 0:
    print(f"\n❌ Command failed with return code {result.returncode}")
    sys.exit(result.returncode)

print("\n✅ All commands finished successfully!")
