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

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2021_realistic', '')# for hichem's DeepCore ntuples
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) ) #-1 = all events

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/uscms_data/d3/njh/GraphCore/Ntuplizer_output1.root'
    ),
)

process.options = cms.untracked.PSet(
#   allowUnscheduled = cms.untracked.bool(True),
#   numberOfThreads = cms.untracked.uint32(1),
#   numberOfStreams = cms.untracked.uint32(1),
   wantSummary = cms.untracked.bool(True)
)

process.ntuples = cms.EDProducer('DeepCoreNtuplizer' ,
 ptMin = cms.double(250) ,
 pMin = cms.double(250),
 etaWhereBarrelEnds = cms.double(1.8),
 deltaR = cms.double(0.1),
 vertices = cms.InputTag("offlinePrimaryVertices"),
 pixelClusters=cms.InputTag("siPixelClustersPreSplitting"),#ntuplizer local reco cluster collection
 #pixelClusters=cms.InputTag("siPixelClusters"), #standard reco cluster collection
 cores = cms.InputTag("ak4CaloJets"),
 simTracks= cms.InputTag("g4SimHits"),
 pixelCPE = cms.string( "PixelCPEGeneric" ),
 PixelDigiSimLinkVector = cms.InputTag("simSiPixelDigis"),
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
 ignoreTotal = cms.untracked.int32(1),
 oncePerEventMode = cms.untracked.bool(True)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("OctopiNtuples.root"),
)
  
process.MessageLogger.cerr.threshold = "Info"
process.MessageLogger.debugModules = ["ntuples"]

process.p = cms.Path(process.ntuples) 
