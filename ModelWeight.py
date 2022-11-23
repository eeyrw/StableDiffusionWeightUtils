from collections import OrderedDict
import os
from pathlib import Path
import torch
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--input', '-I', type=str, help='Input file to prune', required = True)
# args = parser.parse_args()
# file = args.input
from terminaltables import AsciiTable


# def prune_it(p):
#     print(f"prunin' in path: {p}")
#     size_initial = os.path.getsize(p)
#     nsd = dict()
#     sd = torch.load(p, map_location="cpu")
#     print(sd.keys())
#     for k in sd.keys():
#         if k != "optimizer_states":
#             nsd[k] = sd[k]
#     else:
#         print(f"removing optimizer states for path {p}")
#     if "global_step" in sd:
#         print(f"This is global step {sd['global_step']}.")
#     if keep_only_ema:
#         sd = nsd["state_dict"].copy()
#         # infer ema keys
#         ema_keys = {k: "model_ema." + k[6:].replace(".", "") for k in sd.keys() if k.startswith('model.')}
#         new_sd = dict()

#         for k in sd:
#             if k in ema_keys:
#                 print(k, ema_keys[k])
#                 new_sd[k] = sd[ema_keys[k]]
#             elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
#                 new_sd[k] = sd[k]

#         assert len(new_sd) == len(sd) - len(ema_keys)
#         nsd["state_dict"] = new_sd
#     else:
#         sd = nsd['state_dict'].copy()
#         new_sd = dict()
#         for k in sd:
#             new_sd[k] = sd[k]
#         nsd['state_dict'] = new_sd

#     fn = f"{os.path.splitext(p)[0]}-pruned.ckpt" if not keep_only_ema else f"{os.path.splitext(p)[0]}-ema-pruned.ckpt"
#     print(f"saving pruned checkpoint at: {fn}")
#     torch.save(nsd, fn)
#     newsize = os.path.getsize(fn)
#     MSG = f"New ckpt size: {newsize*1e-9:.2f} GB. " + \
#           f"Saved {(size_initial - newsize)*1e-9:.2f} GB by removing optimizer states"
#     if keep_only_ema:
#         MSG += " and non-EMA weights"
#     print(MSG)


def extractVAE(state_dict):
    vaeStateDict = {}
    vaeKeys = ['decoder', 'encoder', 'quant_conv', 'post_quant_conv']
    state_dict = state_dict.copy()
    for k in state_dict.keys():
        for keepKey in vaeKeys:
            if k.startswith(keepKey):
                vaeStateDict[k] = state_dict[k]
            elif k.startswith('first_stage_model.'):
                new_k = str.replace(k, 'first_stage_model.', '')
                if new_k.startswith(keepKey):
                    vaeStateDict[new_k] = state_dict[k]
    return vaeStateDict


def loadWeight(weightFilePath):
    weightDict = torch.load(weightFilePath, map_location="cpu")
    if "global_step" in weightDict:
        print(f"This is global step {weightDict['global_step']}.")
    if "epoch" in weightDict:
        print(f"This is epoch {weightDict['epoch']}.")
    with open(weightFilePath+'.txt', 'w') as f:
        f.writelines([k+'\n' for k in weightDict.keys()])
    with open(weightFilePath+'.state_dict.txt', 'w') as f:
        f.writelines([k+'\n' for k in weightDict['state_dict'].keys()])
    return weightDict


def compareWeightTenosr(tensor1, tensor2, eps=1e-4):
    eqType = 'not equal'
    if tensor1.dtype==tensor2.dtype and torch.equal(tensor1, tensor2):
        eqType = 'strict equal'
        diff_min = 0
        diff_max = 0
        diff_mean = 0
    else:
        diff = torch.abs(tensor1-tensor2)
        diff_min = torch.min(diff)
        diff_max = torch.max(diff)
        diff_mean = torch.mean(diff)
        if diff_mean < eps:
            eqType = 'lossen equal'

    compareResultDict = OrderedDict()
    compareResultDict['eqType'] = eqType
    compareResultDict['diffMin'] = '%.5f'%float(diff_min)
    compareResultDict['diffMax'] = '%.5f'%float(diff_max)
    compareResultDict['diffMean'] = '%.5f'%float(diff_mean)
    compareResultDict['tensor1dType'] = str(tensor1.dtype)
    compareResultDict['tensor2dType'] = str(tensor2.dtype)

    return compareResultDict


def compareWeightDict(dict1, dict2, prefixGroupList=['model.diffusion_model','cond_stage_model','first_stage_model','model_ema']):
    dict1KeysSet = set(dict1.keys())
    dict2KeysSet = set(dict2.keys())
    commonKeys = dict1KeysSet and dict2KeysSet
    dict1OnlyKeys = dict1KeysSet - commonKeys
    dict2OnlyKeys = dict2KeysSet - commonKeys
    commonKeys = list(commonKeys)
    commonKeys.sort()
    groupKeyNameDict = {}
    for k in commonKeys:
        startWithKey = 'unknown'
        for groupKey in prefixGroupList:
            if k.startswith(groupKey):
                startWithKey = groupKey
                break
        groupKeyNameDict.setdefault(
            startWithKey, []).append(k)

    groupResultListDict = {}
    for groudKey, actualKeyList in groupKeyNameDict.items():
        fields = None
        for key in actualKeyList:
            resultDict = compareWeightTenosr(dict1[key],dict2[key])
            outputList = [key[len(groudKey):] if groudKey!='unknown' else key] + [value for field,value in resultDict.items()]
            if not fields:
                fields = ['Key']+[field for field,value in resultDict.items()]
            groupResultListDict.setdefault(
                groudKey, [fields]).append(outputList)
        table = AsciiTable(groupResultListDict[groudKey])
        table.inner_row_border = True
        table.title = groudKey
        with open(groudKey+'.compare_result.txt', 'w') as f:
            f.write(table.table)
            f.write('\n')


    

    # with open('compare_result.txt', 'w') as f:
    #     for k in commonKeys:
    #         d1 = dict1[k].to(torch.float32)
    #         d2 = dict2[k].to(torch.float32)
    #         diff = torch.abs(d1-d2)
    #         diff_min = torch.min(diff).float()
    #         diff_max = torch.max(diff).float()
    #         diff_mean = torch.mean(diff).float()
    #         f.write('%s:\n      MIN:%.5f, MAX:%.5f, MEAN:%.5f\n' %
    #                 (k, diff_min, diff_max, diff_mean))
    #         f.write('      dtype1: %s,dtype2:%s\n' %
    #                 (dict1[k].dtype, dict2[k].dtype))


def removeVAE(weightDict):
    newStateDict = dict()
    keepKeys = ['model.diffusion_model', 'cond_stage_model']
    keepKeys_vae = ['decoder', 'encoder', 'quant_conv', 'post_quant_conv']
    for k in weightDict['state_dict'].keys():
        needKeep = False
        for keepKey in keepKeys+keepKeys_vae:
            if k.startswith(keepKey):
                needKeep = True
                break
        if not needKeep:
            continue
        newStateDict[k] = weightDict['state_dict'][k]
    newWeightDict = dict({'state_dict': newStateDict})
    fn = 'F:/2022-11-08T13-38-59_minimal-ram-single-gpu/checkpoints/zwx-no-vae.ckpt'
    torch.save(newWeightDict, fn)
    newsize = os.path.getsize(fn)
    MSG = f"New ckpt size: {newsize*1e-9:.2f} GB."
    print(MSG)


def replaceVAE(rawDict, vaeDict):
    newStateDict = dict()
    keepKeys = ['model.diffusion_model', 'cond_stage_model']
    for k in rawDict['state_dict'].keys():
        needKeep = False
        for keepKey in keepKeys:
            if k.startswith(keepKey):
                needKeep = True
                break
        if not needKeep:
            continue
        newStateDict[k] = rawDict['state_dict'][k]

    keepKeys_vae = ['decoder', 'encoder', 'quant_conv', 'post_quant_conv']
    vaeKeyPrefix = 'first_stage_model.'
    for k in vaeDict['state_dict'].keys():
        needKeep = False
        for keepKey in keepKeys_vae:
            if k.startswith(keepKey):
                needKeep = True
                break
        if not needKeep:
            continue
        newStateDict[vaeKeyPrefix+k] = vaeDict['state_dict'][k]

    newWeightDict = dict({'state_dict': newStateDict})
    return newWeightDict


def blendWeight(rawDict1, rawDict2):
    newStateDict = dict()
    keepKeys = ['model.diffusion_model',
                'cond_stage_model', 'first_stage_model']
    for k in rawDict1['state_dict'].keys():
        needKeep = False
        for keepKey in keepKeys:
            if k.startswith(keepKey):
                needKeep = True
                break
        if not needKeep:
            continue
        if k in rawDict2['state_dict'].keys():
            newStateDict[k] = (rawDict1['state_dict'][k] +
                               rawDict2['state_dict'][k])/2
        else:
            newStateDict[k] = rawDict1['state_dict'][k]
        newStateDict[k].to(torch.float16)

    newWeightDict = dict({'state_dict': newStateDict})
    return newWeightDict


if __name__ == "__main__":
    weight1Dict = loadWeight(r"D:\weight\Anything-V3.0-pruned.ckpt")
    weight2Dict = loadWeight(r"D:\weight\Anything-V3.0.ckpt")
    # newWeight = replaceVAE(weight1Dict,weight2Dict)
    # newWeight = blendWeight(weight1Dict,weight2Dict)
    # torch.save(newWeight,r'F:/Anything-hd-15.ckpt')
    #vaeWeight1 = extractVAE(weight1Dict['state_dict'])
    #vaeWeight2 = extractVAE(weight2Dict['state_dict'])
    compareWeightDict(weight1Dict['state_dict'], weight2Dict['state_dict'])
    # removeVAE(loadWeight(r"F:/2022-11-08T13-38-59_minimal-ram-single-gpu/checkpoints/zwx.ckpt"))

    # newStateDict = dict()
    # keepKeys = ['model.diffusion_model','cond_stage_model']
    # keepKeys_vae = ['decoder','encoder','quant_conv','post_quant_conv']
    # for k in weightDict['state_dict'].keys():
    #     needKeep = False
    #     for keepKey in keepKeys+keepKeys_vae:
    #         if k.startswith(keepKey):
    #             needKeep = True
    #             break
    #     if not needKeep:
    #         continue
    #     newStateDict[k] = weightDict['state_dict'][k]
    # newWeightDict = dict({'state_dict':newStateDict})
    # fn = 'F:/2022-11-08T13-38-59_minimal-ram-single-gpu/checkpoints/zwx-no-vae.ckpt'
    # torch.save(newWeightDict, fn)
    # newsize = os.path.getsize(fn)
    # MSG = f"New ckpt size: {newsize*1e-9:.2f} GB. " + \
    #       f"Saved {(size_initial - newsize)*1e-9:.2f} GB"
    # print(MSG)
