import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

## process.GlobalTag.globaltag="94X_mc2017_realistic_v10"
#process.GlobalTag.globaltag = "120X_mcRun3_2021_realistic_v6" ## Updating global tag since we are using run 3 2021 mc
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2023_realistic', '')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) ) #-1 = tutti (numero edi eventi)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/uscms_data/d3/njh/GraphCore/Ntuplizer_output1.root'
    ),
)

process.options = cms.untracked.PSet(
   allowUnscheduled = cms.untracked.bool(True),
   numberOfThreads = cms.untracked.uint32(1),
   numberOfStreams = cms.untracked.uint32(1),
   wantSummary = cms.untracked.bool(True)
)

process.ntuples = cms.EDProducer('DeepCoreNtuplizer' ,
 ptMin = cms.double(500) ,
 pMin = cms.double(0),
 deltaR = cms.double(0.1),
 barrelTrain =cms.bool(True),
 endcapTrain =cms.bool(False),
 fullTrain =cms.bool(False),
  
 vertices = cms.InputTag("offlinePrimaryVertices"),
 #pixelClusters=cms.InputTag("siPixelClustersPreSplitting"),
 pixelClusters=cms.InputTag("siPixelClusters"),
 cores = cms.InputTag("ak4CaloJets"),
 centralMIPCharge = cms.double(18000.0),
 chargeFractionMin = cms.double(2),
 simTracks= cms.InputTag("g4SimHits"),
 simHit= cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
 simHitEC= cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
 pixelCPE = cms.string( "PixelCPEGeneric" ),
 PixelDigiSimLinkVector = cms.InputTag("simSiPixelDigis"),
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
 ignoreTotal = cms.untracked.int32(1),
 oncePerEventMode = cms.untracked.bool(True)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("GraphCoreNtuples.root"),
    closeFileFast = cms.untracked.bool(True)
  )
  
process.MessageLogger.cerr.threshold = "Info"
process.MessageLogger.debugModules = ["ntuples"]

process.p = cms.Path(process.ntuples) 
# 500 is the goodone (for barel training) #1000 is the tested one, with p cut insted of pt
