CONFIG={
    "RemoveTrivialDrugs":False,
    "RemoveTerminated":False,
    "SetTerminatedToZero":False,
    "Model":"XGB",
    "Validate":"CV", #CV or CV
    "SampleWeight":'DIINV',
    "Impute":"MEAN", # MEAN, XGB
    "PosScale":"1",
    "TargetRatio":"0.11",
    "UseNumeric":True,
    "Seed":10,
    "FixYear":True, # if fix 1819 
    "DrugindiMax":True, # if perform drug-indi pair indication
}
