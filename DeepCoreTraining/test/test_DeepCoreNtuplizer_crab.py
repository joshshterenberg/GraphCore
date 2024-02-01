from CRABClient.UserUtilities import config,getUsernameFromCRIC
config = config()

config.General.requestName = 'OctopiNtuplesQCDJan31'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'test_DeepCoreNtuplizer.py'
config.JobType.numCores=1
config.JobType.maxMemoryMB=2000


#config.Data.inputDataset='/TT_TuneCP5_13p6TeV_powheg-pythia8/phys_tracking-DeepCoreNtuplizerInput-be26fda57c0e59b5bf1acceb70010d98/USER'
config.Data.inputDBS = 'phys03' 
config.Data.userInputFiles = open('DeepCoreFiles.txt').readlines()
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
NJOBS = 1
config.Data.totalUnits = -1 #config.Data.unitsPerJob * NJOBS


config.Data.publication = False
#config.Data.outputDatasetTag = 'OctopiNtuples'
config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromCRIC())
config.Site.storageSite = 'T3_US_FNALLPC' 
