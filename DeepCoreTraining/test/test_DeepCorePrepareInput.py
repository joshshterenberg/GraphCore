import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

## process = cms.Process('TRAINING',eras.Run2_2017) #barrel trained with eras.Run2_2017, endcap trained with eras.Run2_2018
process = cms.Process('TRAINING',eras.Run3) ## changing era for run 3 since we're using run 3 mc

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
##process.load('HLTrigger.Configuration.HLT_2018v32_cff') 
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring([
      #'root://cms-xrd-global.cern.ch//store/mc/QCD_Pt_1800to2400_TuneCP5_14TeV_pythia8/Run3Winter20DRMiniAOD-FlatPU0to80ALCARECO_110X_mcRun3_2021_realistic_v6-v2/AODSIM'
      #'root://cms-xrd-global.cern.ch//store/mc/RunIISummer17DRPremix/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/AODSIM/92X_upgrade2017_realistic_v10-v5/90000/00043120-439C-E711-9C18-002590FD030A.root' #barrel example
    # 'root://cms-xrd-global.cern.ch//store/mc/RunIIFall17DRStdmix/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/AODSIM/NoPU_94X_mc2017_realistic_v11-v2/20000/0053DFD7-337B-EA11-9E55-002590E39F36.root' #endcap example
	'root://cms-xrd-global.cern.ch//store/mc/Run3Winter23Reco/TT_TuneCP5_13p6TeV_powheg-pythia8/AODSIM'
	]),
    secondaryFileNames = cms.untracked.vstring([
      #'root://cms-xrd-global.cern.ch//store/mc/RunIISummer17GS/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/GEN-SIM/92X_upgrade2017_realistic_v10-v1/70000/100B4392-4586-E711-95F8-0CC47A4D76C0.root', 
      #'root://cms-xrd-global.cern.ch//store/mc/RunIISummer17GS/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/GEN-SIM/92X_upgrade2017_realistic_v10-v1/70000/465BA88A-4386-E711-BBCD-0025905B8612.root',
      #'root://cms-xrd-global.cern.ch//store/mc/RunIISummer17GS/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/GEN-SIM/92X_upgrade2017_realistic_v10-v1/70000/A2A9FD63-5186-E711-B0DC-0CC47A4D769A.root'
      ##'root://cms-xrd-global.cern.ch//store/mc/RunIISummer17GS/QCD_Pt_1800to2400_TuneCUETP8M1_13TeV_pythia8/GEN-SIM/92X_upgrade2017_realistic_v10-v1/50000/347DE057-508C-E711-9FF0-00259020080C.root' #barrel example
        # 'root://cms-xrd-global.cern.ch//store/mc/RunIIFall17DRStdmix/UBGGun_E-1000to7000_Eta-1p2to2p1_13TeV_pythia8/GEN-SIM-DIGI-RAW/NoPU_94X_mc2017_realistic_v11-v2/20000/000AD3DA-BA78-EA11-A244-AC1F6B8AC0CE.root' #endcap example
#	'root://cms-xrd-global.cern.ch//store/mc/Run3Winter23Reco/TT_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM-RECO'
	'root://cms-xrd-global.cern.ch//store/mc/Run3Winter23wmLHEGS/TT_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM'
	]) 
)

process.options = cms.untracked.PSet(
   allowUnscheduled = cms.untracked.bool(True),
   numberOfThreads = cms.untracked.uint32(8),
   numberOfStreams = cms.untracked.uint32(8),
   wantSummary = cms.untracked.bool(True)
)


process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:Ntuplizer_step1.root'),
    outputCommands = cms.untracked.vstring(["drop *"]),
    splitLevel = cms.untracked.int32(0)
)

process.RECOSIMoutput.outputCommands.append("keep SimTracks_g4SimHits_*_*")
process.RECOSIMoutput.outputCommands.append("keep SimVertexs_g4SimHits_*_*")
process.RECOSIMoutput.outputCommands.append("keep PSimHits_g4SimHits_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_ak4CaloJets_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_offlinePrimaryVertices_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_*iPixelCluster*_*_*")
process.RECOSIMoutput.outputCommands.append("keep *_simSiPixelDigis_*_*")


# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
##process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '') #barrel trained with "auto:phase1_2017_realistic", endcap trained with "auto:phase1_2018_realistic"
#process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase1_2021_realistic','') ## Using 2021 since we are using run 3 mc 
process.GlobalTag = GlobalTag(process.GlobalTag, '126X_mcRun3_2023_forPU65_v1', '') #Using 2023 PU 65

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.localreco)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend([process.raw2digi_step,process.reconstruction_step])
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
