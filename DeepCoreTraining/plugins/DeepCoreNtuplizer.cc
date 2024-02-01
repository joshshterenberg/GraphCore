// -*- C++ -*-
//
// Package:    RecoTracker/DeepCoreTraining
// Class:      DeepCoreNtuplizer
//
/**\class DeepCoreNtuplizer DeepCoreNtuplizer.cc RecoTracker/DeepCoreTraining/plugins/DeepCoreNtuplizer.cc

Description: Save inner tracker pixels in the cores of jets as pointclouds for OCToPi training

*/
//
// Adapted from Valerio Bertacchi's DeepCore Ntuplizer
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"



#include <boost/range.hpp>
#include <boost/foreach.hpp>
#include "boost/multi_array.hpp"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"



#include "TTree.h"

#include <tuple>
//
// class declaration
//

class DeepCoreNtuplizer : public edm::stream::EDProducer<> {
    public:
        explicit DeepCoreNtuplizer(const edm::ParameterSet&);
        ~DeepCoreNtuplizer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

        struct TrackAndState
        {
            TrackAndState(const reco::Track *aTrack, TrajectoryStateOnSurface aState) :
                track(aTrack), state(aState) {}
            const reco::Track*      track;
            TrajectoryStateOnSurface state;
        };

        template<typename Cluster>
            struct ClusterWithTracks
            {
                ClusterWithTracks(const Cluster &c) : cluster(&c) {}
                const Cluster* cluster;
                std::vector<TrackAndState> tracks;
            };

        typedef ClusterWithTracks<SiPixelCluster> SiPixelClusterWithTracks;

        typedef boost::sub_range<std::vector<SiPixelClusterWithTracks> > SiPixelClustersWithTracks;

        TFile* DeepCoreNtuplizer_out;
        TTree* DeepCoreNtuplizerTree;
        int eventID;
        
        float caloJetP,caloJetPt,caloJetEta,caloJetPhi;
        std::vector<float> pixelXvec,pixelYvec,pixelRvec,pixelZvec,pixelUvec,pixelVvec,pixelEtavec,pixelPhivec,pixelChargevec, pixelSimTrackPtvec, pixelSimTrackEtavec, pixelSimTrackPhivec;
        std::vector<int> pixelTrackerLayervec,pixelSimTrackIDvec, pixelSimTrackPdgvec;


    private:
        void produce(edm::Event&, const edm::EventSetup&) override;

        // ----------member data ---------------------------

        std::string propagatorName_;

        //Migration to ESGetToken from ESHandle
        edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>          magfield_;
        edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;
        edm::ESGetToken<Propagator,TrackingComponentsRecord>             propagator_;

        edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> parEstHandle;
        edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoHandle; 

        edm::EDGetTokenT<std::vector<reco::Vertex> > vertices_;
        edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusters_;
        std::vector<SiPixelClusterWithTracks> allSiPixelClusters;
        std::map<uint32_t, SiPixelClustersWithTracks> siPixelDetsWithClusters;
        edm::Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
        edm::EDGetTokenT<edm::View<reco::Candidate> > cores_;
        edm::EDGetTokenT<std::vector<SimTrack> > simtracksToken;


        const std::vector<SimTrack> *simtracksVector;
        edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > pdslToken;
        edm::Handle< edm::DetSetVector<PixelDigiSimLink> > PDSLContainer;
        const edm::DetSetVector<PixelDigiSimLink> *pixelSimLinks;


        double ptMin_;
        double pMin_;
        double etaWhereBarrelEnds_;
        double deltaR_;
        std::string pixelCPE_;

        std::pair<int,int> local2Pixel(double, double, const GeomDet*);
        LocalPoint pixel2Local(int, int, const GeomDet*);
        std::tuple<int,int,float,float,float> getOneSimTrackFromPixel(SiPixelCluster::Pixel pixel, unsigned int detId);

};

DeepCoreNtuplizer::DeepCoreNtuplizer(const edm::ParameterSet& iConfig) :
    magfield_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
    geometryToken_(esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()),
    propagator_(esConsumes<Propagator,TrackingComponentsRecord>()),

    parEstHandle(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")))),
    tTopoHandle(esConsumes<TrackerTopology, TrackerTopologyRcd>()),

    vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
    pixelClusters_(consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
    cores_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("cores"))),
    simtracksToken(consumes<std::vector<SimTrack> >(iConfig.getParameter<edm::InputTag>("simTracks"))),
    pdslToken(consumes<edm::DetSetVector<PixelDigiSimLink> >(iConfig.getParameter<edm::InputTag>("PixelDigiSimLinkVector"))),
    ptMin_(iConfig.getParameter<double>("ptMin")),
    pMin_(iConfig.getParameter<double>("pMin")),
    etaWhereBarrelEnds_(iConfig.getParameter<double>("etaWhereBarrelEnds")),
    deltaR_(iConfig.getParameter<double>("deltaR")),
    pixelCPE_(iConfig.getParameter<std::string>("pixelCPE"))
{

    //  usesResource("TFileService");
    edm::Service<TFileService> fileService;

    DeepCoreNtuplizerTree= fileService->make<TTree>("tree","tree");
    DeepCoreNtuplizerTree->SetAutoSave(0);
    
    //Global variables
    DeepCoreNtuplizerTree->Branch("event",&eventID);
    DeepCoreNtuplizerTree->Branch("caloJetP",&caloJetP);
    DeepCoreNtuplizerTree->Branch("caloJetPt",&caloJetPt);
    DeepCoreNtuplizerTree->Branch("caloJetEta",&caloJetEta);
    DeepCoreNtuplizerTree->Branch("caloJetPhi",&caloJetPhi);

    //pixel variables

    DeepCoreNtuplizerTree->Branch("pixelR",&pixelRvec);
    DeepCoreNtuplizerTree->Branch("pixelPhi",&pixelPhivec);
    DeepCoreNtuplizerTree->Branch("pixelZ",&pixelZvec);
    
    DeepCoreNtuplizerTree->Branch("pixelEta",&pixelEtavec);
    
    //DeepCoreNtuplizerTree->Branch("pixelX",&pixelXvec);
    //DeepCoreNtuplizerTree->Branch("pixelY",&pixelYvec);
    //Conformal coordinates instead of x,y
    DeepCoreNtuplizerTree->Branch("pixelU",&pixelUvec);
    DeepCoreNtuplizerTree->Branch("pixelV",&pixelVvec);

    DeepCoreNtuplizerTree->Branch("pixelCharge",&pixelChargevec);
    DeepCoreNtuplizerTree->Branch("pixelTrackerLayer",&pixelTrackerLayervec);
    
    //truth labeling
    DeepCoreNtuplizerTree->Branch("pixelSimTrackID",&pixelSimTrackIDvec);
    //extra truth info not needed for training
    //DeepCoreNtuplizerTree->Branch("pixelSimTrackPdg",&pixelSimTrackPdgvec);
    //DeepCoreNtuplizerTree->Branch("pixelSimTrackPt",&pixelSimTrackPtvec);
    //DeepCoreNtuplizerTree->Branch("pixelSimTrackEta",&pixelSimTrackEtavec);
    //DeepCoreNtuplizerTree->Branch("pixelSimTrackPhi",&pixelSimTrackPhivec);




}

DeepCoreNtuplizer::~DeepCoreNtuplizer() {
    // do anything here that needs to be done at destruction time
    // (e.g. close files, deallocate resources etc.)
    //
    // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
#define foreach BOOST_FOREACH

// ------------ method called to produce the data  ------------
void DeepCoreNtuplizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    using namespace edm;

    eventID= iEvent.id().event();

    const GlobalTrackingGeometry* geometry_ = &iSetup.getData(geometryToken_);


    iEvent.getByToken(pixelClusters_, inputPixelClusters);
    allSiPixelClusters.clear(); siPixelDetsWithClusters.clear();
    allSiPixelClusters.reserve(inputPixelClusters->dataSize()); // this is important, otherwise push_back invalidates the iterators

    edm::Handle<std::vector<SimTrack> > simtracks;
    iEvent.getByToken(simtracksToken, simtracks);

    Handle<std::vector<reco::Vertex> > vertices;
    iEvent.getByToken(vertices_, vertices);



    Handle<edm::View<reco::Candidate> > cores;
    iEvent.getByToken(cores_, cores);

    iEvent.getByToken(pdslToken, PDSLContainer);
    pixelSimLinks = PDSLContainer.product();

    
    
    simtracksVector = simtracks.product();

    const PixelClusterParameterEstimator* parEst = &iSetup.getData(parEstHandle);  

    const TrackerTopology* tTopo = &iSetup.getData(tTopoHandle);


    auto output = std::make_unique<edmNew::DetSetVector<SiPixelCluster>>();



    //Go OCToPi

    for (unsigned int jetIdx = 0; jetIdx < cores->size(); jetIdx++) { //loop jet
        const reco::Candidate& jet = (*cores)[jetIdx];
        if ((jet.pt() > ptMin_ && std::abs(jet.eta())< etaWhereBarrelEnds_) || (jet.p()>pMin_ && std::abs(jet.eta())>etaWhereBarrelEnds_ && std::abs(jet.eta())<2.5 )) { 
            caloJetP = jet.p();
            caloJetPt = jet.pt();
            caloJetEta = jet.eta();
            caloJetPhi = jet.phi();

            //std::cout << "Jet! " << jet.eta() << " " << jet.phi() << std::endl;
            const reco::Vertex& jetVertex = (*vertices)[0];
            GlobalPoint jetVertexPoint(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
            //std::cout << "Jet Vertex Position " << jetVertexPoint.eta() << " " <<jetVertexPoint.phi() << " z=" << jetVertexPoint.z() << std::endl;
            GlobalVector jetDirection(jet.px(), jet.py(), jet.pz());

            int nPixels=0;
            for ( edmNew::DetSetVector<SiPixelCluster>::const_iterator clusterDetSetVecItr = inputPixelClusters->begin(); clusterDetSetVecItr != inputPixelClusters->end(); clusterDetSetVecItr++) { //loop detsets of clusters
                const edmNew::DetSet<SiPixelCluster>& DetSetOfClusters = *clusterDetSetVecItr;
                const GeomDet* det = geometry_->idToDet(DetSetOfClusters.id()); 
                for (auto cluster = DetSetOfClusters.begin(); cluster != DetSetOfClusters.end(); cluster++) {//loop clusters (finally)

                    GlobalPoint clusterPos = det->surface().toGlobal(parEst->localParametersV(*cluster,(*geometry_->idToDetUnit(DetSetOfClusters.id())))[0].first);//cluster position, as estimated from pixels. Needed to compare to jet direction
                    GlobalVector clusterDirection = clusterPos - jetVertexPoint;
                    if (Geom::deltaR(jetDirection, clusterDirection) < deltaR_) {
                        //std::cout << "\tfound cluster" << clusterPos.eta() << " " << clusterPos.phi() << " " << clusterPos.z() << " deltaR=" << Geom::deltaR(jetDirection, clusterDirection) << std::endl; 
                        //get global positions of every pixel in cluster
                        for(int pixel_i=0; pixel_i<cluster->size(); pixel_i++){
                            SiPixelCluster::Pixel pixel=cluster->pixel(pixel_i);
                            LocalPoint pixelLocalPoint = pixel2Local(pixel.x,pixel.y,det);
                            GlobalPoint pixelGlobalPoint = det->toGlobal(pixelLocalPoint);
                            float pixelX = pixelGlobalPoint.x();
                            float pixelY = pixelGlobalPoint.y();
                            if(fabs(pixelGlobalPoint.phi())>0) nPixels++;
                            
                            int pixelLayer = tTopo->layer(det->geographicalId());
                            if(det->geographicalId().subdetId()==PixelSubdetector::PixelEndcap) {
                                pixelLayer+=4; //endcap layer counting = 5,6,7
                            }
                            //std::cout << "\t\tpixel: " << pixelGlobalPoint.eta() << " " << pixelGlobalPoint.phi() << " " << pixelGlobalPoint.z() << " " << pixelLayer << "\t" << (pixel.adc)/(float) (14000) << std::endl;
                           
                            
                            //TRUTH: for a given pixel, which simtrack left the most charge in it?
                            std::tuple<int,int,float,float,float> simTrackIdPdgPtEtaPhi;
                            simTrackIdPdgPtEtaPhi = DeepCoreNtuplizer::getOneSimTrackFromPixel( pixel, DetSetOfClusters.id());

                            //FILL
                            pixelXvec.push_back(pixelX);
                            pixelYvec.push_back(pixelY);
                            pixelRvec.push_back(sqrt(pixelX*pixelX+pixelY*pixelY));
                            pixelZvec.push_back(pixelGlobalPoint.z());
                            pixelEtavec.push_back(pixelGlobalPoint.eta());
                            pixelPhivec.push_back(pixelGlobalPoint.phi());
                            pixelTrackerLayervec.push_back(pixelLayer);
                            pixelChargevec.push_back((pixel.adc)/(float)(14000));
                            pixelUvec.push_back(pixelX/(pixelX*pixelX+pixelY*pixelY));
                            pixelVvec.push_back(pixelY/(pixelX*pixelX+pixelY*pixelY));


                            pixelSimTrackIDvec.push_back( std::get<0>(simTrackIdPdgPtEtaPhi));
                            pixelSimTrackPdgvec.push_back(std::get<1>(simTrackIdPdgPtEtaPhi));
                            pixelSimTrackPtvec.push_back( std::get<2>(simTrackIdPdgPtEtaPhi));
                            pixelSimTrackEtavec.push_back(std::get<3>(simTrackIdPdgPtEtaPhi));
                            pixelSimTrackPhivec.push_back(std::get<4>(simTrackIdPdgPtEtaPhi));
                        }
                    }
                }
            }
            //std::cout << "tot pixels: " << nPixels << std::endl; //50-300 pixels within deltaR=0.1. Too many?
            
            if(nPixels>1) DeepCoreNtuplizerTree->Fill(); //a single pixel breaks contrastive loss function
            pixelXvec.clear();
            pixelYvec.clear();
            pixelZvec.clear();
            pixelEtavec.clear();
            pixelPhivec.clear();
            pixelTrackerLayervec.clear();
            pixelChargevec.clear();
            pixelRvec.clear();
            pixelUvec.clear();
            pixelVvec.clear();
            
            pixelSimTrackIDvec.clear();
            pixelSimTrackPdgvec.clear();
            pixelSimTrackPtvec.clear();
            pixelSimTrackEtavec.clear();
            pixelSimTrackPhivec.clear();
        }
    }
}


std::tuple<int,int,float,float,float> DeepCoreNtuplizer::getOneSimTrackFromPixel(SiPixelCluster::Pixel pixel, unsigned int detId){
    
    std::tuple<int,int,float,float,float> simTrackIdPdgPtEtaPhi;
    
    std::get<0>(simTrackIdPdgPtEtaPhi)=-99;
    std::get<1>(simTrackIdPdgPtEtaPhi)=-99;
    std::get<2>(simTrackIdPdgPtEtaPhi)=-99;
    std::get<3>(simTrackIdPdgPtEtaPhi)=-99;
    std::get<4>(simTrackIdPdgPtEtaPhi)=-99;
     
    //store simtrack ID with greatest charge in pixel
    float maxChargeFraction=0.0;
    auto firstLink = pixelSimLinks->find(detId);
    if(firstLink != pixelSimLinks->end()){
        auto link_detset = (*firstLink);

        for(auto linkiter : link_detset.data){
            std::pair<int,int> pos = PixelDigi::channelToPixel(linkiter.channel());

            if(pos.first == pixel.x && pos.second == pixel.y){
                for(auto iter = simtracksVector->begin(); iter != simtracksVector->end(); ++iter){
                    if(iter->trackId() == linkiter.SimTrackId() && iter->momentum().Pt()>0.2 && iter->momentum().Pt()<99999){ //cut on P() instead?
                        if(linkiter.fraction()>maxChargeFraction){
                            maxChargeFraction=linkiter.fraction();
                            //maxChargeSimTrackID=iter->trackId();

                            std::get<0>(simTrackIdPdgPtEtaPhi)=iter->trackId();
                            std::get<1>(simTrackIdPdgPtEtaPhi)=iter->type();
                            std::get<2>(simTrackIdPdgPtEtaPhi)=iter->momentum().Pt();
                            std::get<3>(simTrackIdPdgPtEtaPhi)=iter->momentum().eta();
                            std::get<4>(simTrackIdPdgPtEtaPhi)=iter->momentum().phi();

                        }
                    }
                }
            }
        }
    }
    return simTrackIdPdgPtEtaPhi;
}

std::pair<int,int> DeepCoreNtuplizer::local2Pixel(double locX, double locY, const GeomDet* det){
    LocalPoint locXY(locX,locY);
    float pixX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).first;
    float pixY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).second;
    std::pair<int, int> out(pixX,pixY);
    return out;
}

LocalPoint DeepCoreNtuplizer::pixel2Local(int pixX, int pixY, const GeomDet* det){
    float locX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localX(pixX);
    float locY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localY(pixY);
    LocalPoint locXY(locX,locY);
    return locXY;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DeepCoreNtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
    desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
    desc.add<edm::InputTag>("cores", edm::InputTag("ak4CaloJets"));
    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
    desc.add<double>("ptMin", 250);
    desc.add<double>("pMin", 250);
    desc.add<double>("etaWhereBarrelEnds", 1.8);
    desc.add<double>("deltaR", 0.1);
    desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
    desc.setUnknown();
    descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepCoreNtuplizer);
