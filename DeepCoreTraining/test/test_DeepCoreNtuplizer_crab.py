from CRABClient.UserUtilities import config,getUsernameFromCRIC
config = config()

config.General.requestName = 'DeepCoreNtuplizerJul11'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
##config.JobType.psetName = 'NNClustSeedInputSimHit_config.py'
config.JobType.psetName = 'test_DeepCoreNtuplizer.py' ## correct python file
config.JobType.numCores=8
config.JobType.maxMemoryMB=20000


#config.Data.inputDataset='/store/group/phys_tracking/Soohyuny/DeepCoreNtuplizer/TT_TuneCP5_13p6TeV_powheg-pythia8/DeepCoreNtuplizerInput/'
config.Data.inputDataset='/TT_TuneCP5_13p6TeV_powheg-pythia8/phys_tracking-DeepCoreNtuplizerInput-be26fda57c0e59b5bf1acceb70010d98/USER'
## config.Data.inputDataset = '/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/arizzi-TrainJetCoreAll-ddeeece6d9d1848c03a48f0aa2e12852/USER' #barrel input
## config.Data.inputDataset = '/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/211013_194728/0000/output'
#config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/hboucham-DeepCoreNtuplizerInput-6d87375326d3f4c3992ae44982f1a1bf/USER'
#config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/hboucham-DeepCoreNtuplizerInput-e2166fcccba2bba0572671bbca4b6777/USER'
#config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/hboucham-DeepCoreNtuplizerInput-6d87375326d3f4c3992ae44982f1a1bf/USER'
## config.Data.userInputFiles = ['/eos/uscms/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/211013_194728/0000/Ntuplizer_output1.root']
## config.Data.userInputFiles = ['/eos/uscms/store/user/hichemb/RelValQCD_Pt_1800_2400_14/DeepCoreNtuplizerInput/211013_194728/0000/Ntuplizer_output1.root']
# config.Data.inputDataset = '/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/vbertacc-DeepCoreTrainingSampleEC_all-3b4718db5896f716d6af32b678bbc9f2/USER' #endcap input
config.Data.inputDBS = 'phys03' ##'phys03' ## might need to change to global?
config.Data.splitting = 'EventAwareLumiBased'
config.Data.unitsPerJob = 1000
NJOBS = 5000
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS

#config.Data.outLFNDirBase = '/store/user/%s/DeepCoreTrainingSample' % (getUsernameFromCRIC())

config.Data.publication = True
config.Data.outputDatasetTag = 'DeepCoreTrainingSample'
##config.Data.totalUnits = config.Data.unitsPerJob*10000

#config.Site.storageSite = "T2_IT_Pisa
#config.Site.storageSite = "T3_US_FNALLPC"
config.Site.storageSite = 'T2_CH_CERN'
config.Data.outLFNDirBase = '/store/group/phys_tracking/Soohyuny/DeepCoreNtuplizer/'
