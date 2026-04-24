import subprocess
import sys



# #       \"P0123\"  \"P0408\"        \"P0410\"
##      P1026   
# P1025   P0613   P0323   P0305   P0202   P0104
#

commands = [
    # "python base_SC_attack.py --dataset \"P1026\"",
    # "python DL_SC_attack.py --com-time 100 --client 5 --epochs 10 --dataset \"P1026\"",
    "python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset \"P1026\" --epsilon 10",

    # "python base_SC_attack.py --dataset \"P0410\"",
    # "python DL_SC_attack.py --com-time 100 --client 5 --epochs 10 --dataset \"P0410\"",
    "python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset \"P0410\" --epsilon 10",



    # "python base_SC_attack.py --dataset \"P0408\"",
    # "python DL_SC_attack.py --com-time 100 --client 5 --epochs 10 --dataset \"P0408\"",
    # "python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset \"P0408\" --epsilon 10",

    # "python base_SC_attack.py --dataset \"P0123\"",
    # "python DL_SC_attack.py --com-time 100 --client 5 --epochs 10 --dataset \"P0123\"",
    # "python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset \"P0123\" --epsilon 10",
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








# python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset "P1026" --epsilon 10

# est model saved from Epoch 45
# Best metrics: {'ARI': 0.34617335191839105, 'NMI': 0.6367038830309272, 'CA': 0.4046928916494134, 'JI': 0.2477118741355232, 'CS': 0.8463838039224311, 'HS': 0.5102872505036051, 'VMS': 0.
               
# python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset "P0410" --epsilon 10




# python DL_SC_attack.py --com-time 100 --client 5 --epochs 10 --dataset "P0408"

# python base_SC_attack.py --dataset "P0408"

# python base_SC_attack.py --dataset "P0410"



# Best model saved from Epoch 7
# Best metrics: {'ARI': 0.48471268417805335, 'NMI': 0.666438042198476, 'CA': 0.5367158854424735, 'JI': 0.38162599794978613, 'CS': 0.5988037938276469, 'HS': 0.7512961378015591, 'VMS': 0.6664380421984761}


# #  把kmeans的聚类数量设置给关掉！！！ 
# python base_SC_attack.py --dataset "P0408"



# ### python DL_SC_attack.py --com-time 100 --client 5 --epochs 10 --dataset "P1026"
# # python PFDL_SC_attack.py --com-time 1 --client 2 --epochs 2 --delta 0.001 --l2_clip 2 --dataset "P1026" --epsilon 10

# # \"P0123\"       yes
# # \"P0408\"       yes2
# # \"P0410\"       yes3
# # commands = [
# #     "python base_SC_attack.py --dataset \"P1026\"", yes1
# #     "python base_SC_attack.py --dataset \"P1025\"", no
# #     "python base_SC_attack.py --dataset \"P0613\"", no
# #     "python base_SC_attack.py --dataset \"P0323\"", no
# #     "python base_SC_attack.py --dataset \"P0305\"", # 不行，图不好看
# #     "python base_SC_attack.py --dataset \"P0202\"", # 不行，图不好看
# #     "python base_SC_attack.py --dataset \"P0104\"", # 不行，图不好看

# # ]


# python base_SC_attack.py --dataset "P1026"




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="VGAE pipeline for scRNA-seq")
#     parser.add_argument("--dataset", type=str, default="yan", help="dataset name (without .h5)")
#     parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
#     parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

#     parser.add_argument("--gnn-hidden", type=int, default=1024)
#     parser.add_argument("--gnn-latent", type=int, default=128)
#     parser.add_argument("--graph-method", type=str, default="pearson", )    # choices=["pearson","cosine","gaussian"]

#     parser.add_argument("--num-neg", type=int, default=10)
#     parser.add_argument("--temperature", type=float, default=0.2)



# python base_SC_attack.py --dataset "P0123"


# Best model saved from Epoch 72
# Best metrics: {'ARI': 0.6857274387699074, 'NMI': 0.7350637369908839, 'CA': 0.668693009118541, 'JI': 0.5728866524574927, 'CS': 0.6645423471524339, 'HS': 0.8223295408555145, 'VMS': 0.735063736990884}


# python base_SC_attack.py --dataset "P1026"  --lr 0.01
# 差评
# python base_SC_attack.py --dataset "P1026"  --lr 0.0001
# 差评

# python base_SC_attack.py --dataset "P0123"  --num-neg 15
# Best model saved from Epoch 95
# Best metrics: {'ARI': 0.55218715576047, 'NMI': 0.723489222092967, 'CA': 0.6531710079275198, 'JI': 0.40404457306624886, 'CS': 0.7206326993610834, 'HS': 0.7263684809433882, 'VMS': 0.723489222092967}


# python base_SC_attack.py --dataset "P1026"  --epochs 150
# Best model saved from Epoch 137
# Best metrics: {'ARI': 0.575970450816653, 'NMI': 0.735705241337553, 'CA': 0.6509060022650056, 'JI': 0.426493537133165, 'CS': 0.7309091055244684, 'HS': 0.7405647362137505, 'VMS': 0.735705241337553}


# python base_SC_attack.py --dataset "P1026"  --epochs 300
# Best model saved from Epoch 287
# Best metrics: {'ARI': 0.6030837389550759, 'NMI': 0.7441776440307369, 'CA': 0.6913929784824462, 'JI': 0.4523700181083167, 'CS': 0.733468900620814, 'HS': 0.7552037186722601, 'VMS': 0.744177644030737}


# python base_SC_attack.py --dataset "P1026"  --epochs 150 --lr 0.01
# 不收敛
# python base_SC_attack.py --dataset "P1026"  --epochs 150 --lr 0.005
# 不收敛
# python base_SC_attack.py --dataset "P1026"  --epochs 150 --lr 0.002
# 不收敛


# python base_SC_attack.py --dataset "P1026"  --epochs 150 --lr 0.002
# 0.33        但是，图好看


# python base_SC_attack.py --dataset "P1026"  --num-neg 15  --epochs 300
# Best model saved from Epoch 257
# Best metrics: {'ARI': 0.5981166696418686, 'NMI': 0.7408297590868851, 'CA': 0.6782276330690826, 'JI': 0.44823830659499725, 'CS': 0.7343251401061458, 'HS': 0.7474506431820488, 'VMS': 0.740829759086885}


# python base_SC_attack.py --dataset "P1026"  --epochs 150



# python base_SC_attack.py --dataset "P0408"  --epochs 400
# Best model saved from Epoch 7
# Best metrics: {'ARI': 0.48471268417805335, 'NMI': 0.666438042198476, 'CA': 0.5367158854424735, 'JI': 0.38162599794978613, 'CS': 0.5988037938276469, 'HS': 0.7512961378015591, 'VMS': 0.6664380421984761}


# python PFDL_SC_attack.py --com-time 100 --client 5 --epochs 10 --delta 0.001 --l2_clip 2 --dataset "P0408" --epsilon 10


