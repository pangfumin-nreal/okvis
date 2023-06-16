//
// Created by pang on 23-6-16.
//
#include <glog/logging.h>
#include <okvis/Estimator.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Remove an observation from a landmark.
    bool Estimator::removeObservation(::ceres::ResidualBlockId residualBlockId) {
        const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
        const uint64_t landmarkId = parameters.at(1).first;
        // remove in landmarksMap
        MapPoint& mapPoint = landmarksMap_.at(landmarkId);
        for(std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.begin();
            it!= mapPoint.observations.end(); ){
            if(it->second == uint64_t(residualBlockId)){

                it = mapPoint.observations.erase(it);
            } else {
                it++;
            }
        }
        // remove residual block
        mapPtr_->removeResidualBlock(residualBlockId);
        return true;
    }

// Remove an observation from a landmark, if available.
    bool Estimator::removeObservation(uint64_t landmarkId, uint64_t poseId,
                                      size_t camIdx, size_t keypointIdx) {
        if(landmarksMap_.find(landmarkId) == landmarksMap_.end()){
            for (PointMap::iterator it = landmarksMap_.begin(); it!= landmarksMap_.end(); ++it) {
                LOG(INFO) << it->first<<", no. obs = "<<it->second.observations.size();
            }
            LOG(INFO) << landmarksMap_.at(landmarkId).id;
        }
        OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                              "landmark not added");

        okvis::KeypointIdentifier kid(poseId,camIdx,keypointIdx);
        MapPoint& mapPoint = landmarksMap_.at(landmarkId);
        std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.find(kid);
        if(it == landmarksMap_.at(landmarkId).observations.end()){
            return false; // observation not present
        }

        // remove residual block
        mapPtr_->removeResidualBlock(reinterpret_cast< ::ceres::ResidualBlockId>(it->second));

        // remove also in local map
        mapPoint.observations.erase(it);

        return true;
    }

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
    template<class T>
    bool vectorContains(const std::vector<T> & vector, const T & query){
        for(size_t i=0; i<vector.size(); ++i){
            if(vector[i] == query){
                return true;
            }
        }
        return false;
    }


    bool Estimator::collectParametersToMargianlize(size_t numKeyframes, size_t numImuFrames,
        std::vector<uint64_t>& paremeterBlocksToBeMarginalized,
        std::vector<bool>& keepParameterBlocks,
        okvis::MapPointVector& removedLandmarks,
        bool& reDoFixation) {

      // keep the newest numImuFrames
      std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
      for(size_t k=0; k<numImuFrames; k++){
        rit++;
        if(rit==statesMap_.rend()){
          // nothing to do.
          return false;
        }
      }

      // distinguish if we marginalize everything or everything but pose
      std::vector<uint64_t> removeFrames;
      std::vector<uint64_t> removeAllButPose;
      std::vector<uint64_t> allLinearizedFrames;
      size_t countedKeyframes = 0;
//  std::cout << "rit->second.isKeyframe: " << rit->second.isKeyframe << " " <<  statesMap_.size() << std::endl;
      while (rit != statesMap_.rend()) {
        if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) {
          removeFrames.push_back(rit->second.id);
        } else {
          countedKeyframes++;
        }
        removeAllButPose.push_back(rit->second.id);
        allLinearizedFrames.push_back(rit->second.id);
        ++rit;// check the next frame
      }

//  std::cout << "removeFrames: " << removeFrames.size() << " " << removeAllButPose.size() << std::endl;
//  for (int i = 0; i < removeAllButPose.size(); i++) {
//    std::cout << "I:  " << i << " " << removeAllButPose.at(i) << std::endl;
//  }


      { // print sliding window info

        std::cout << "*******************" << std::endl;
        std::cout << "-- " << "index" << " \t "
                  << "id"  << " \t " <<  "isKeyframe"
                  << " \t " << "remove"  << " \t " << "remove_1"  << std::endl;
        auto ritr = statesMap_.rbegin();
        int frame_cnt = 0;
        for(; ritr != statesMap_.rend(); ritr++) {
          bool in_removeFrames = false;
          bool in_removeAllButPose = false;
          for (int i = 0; i < removeFrames.size(); i++) {
            if (removeFrames[i] == ritr->second.id) {
              in_removeFrames = true;
              break;
            }
          }

          for (int i = 0; i < removeAllButPose.size(); i++) {
            if (removeAllButPose[i] == ritr->second.id) {
              in_removeAllButPose = true;
              break;
            }
          }

          std::cout << "-- " << frame_cnt << " \t "
                    << ritr->second.id  << " \t " <<  ritr->second.isKeyframe
                    << " \t " << in_removeFrames  << " \t " << in_removeAllButPose  << std::endl;

          frame_cnt++;
        }
      }


      // marginalize everything but pose:
      // Note: camera_imu extrinsic pose not included.
      for(size_t k = 0; k<removeAllButPose.size(); ++k){
        std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);
//    std::cout << "removeAllButPose: " << k << " " << removeAllButPose[k] << " " << it->second.global.size() << std::endl;

        for (size_t i = 0; i < it->second.global.size(); ++i) {
          if (i == GlobalStates::T_WS) {
            continue; // we do not remove the pose here.
          }
          if (!it->second.global[i].exists) {
            continue; // if it doesn't exist, we don't do anything.
          }
          if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if(checkit->second.global[i].exists &&
             checkit->second.global[i].id == it->second.global[i].id){
            continue;
          }

          it->second.global[i].exists = false; // remember we removed

          paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
            it->second.global[i].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
            if(!reprojectionError){   // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
        // add all error terms of the sensor states.
        // Note(pang): only speedbias is margined;
        for (size_t i = 0; i < it->second.sensors.size(); ++i) {
          for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
            for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
              if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) {
//              std::cout << "skip 1 " << i << " " << j << " " << k << std::endl;
                continue; // we do not remove the extrinsics pose here.
              }
              if (!it->second.sensors[i][j][k].exists) {
//              std::cout << "skip 2 " << i << " " << j << " " << k << std::endl;
                continue;
              }
              if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
                ->fixed()) {
//              std::cout << "skip 3 " << i << " " << j << " " << k << std::endl;
                continue;  // we never eliminate fixed blocks.
              }
              std::map<uint64_t, States>::iterator checkit = it;
              checkit++;
              // only get rid of it, if it's different
              if(checkit->second.sensors[i][j][k].exists &&
                 checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
//              std::cout << "skip 4 " << i << " " << j << " " << k << std::endl;
                continue;
              }

//          std::cout << "remove: " << i << " " << j << " " << k << " "
//            << it->second.sensors[i].size() << " " << it->second.sensors[i][j].size() <<  std::endl;

              it->second.sensors[i][j][k].exists = false; // remember we removed
              paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
              keepParameterBlocks.push_back(false);
              ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
                it->second.sensors[i][j][k].id);
              for (size_t r = 0; r < residuals.size(); ++r) {
                std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                  std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                    residuals[r].errorInterfacePtr);
                if(!reprojectionError){   // we make sure no reprojection errors are yet included.
                  marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
                }
              }
            }
          }
        }
      }
      // marginalize ONLY pose now:

      for(size_t k = 0; k<removeFrames.size(); ++k){
        std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

        // schedule removal - but always keep the very first frame.
        //if(it != statesMap_.begin()){
        if(true){ /////DEBUG
          it->second.global[GlobalStates::T_WS].exists = false; // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
          keepParameterBlocks.push_back(false);
        }

        // add remaing error terms
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.global[GlobalStates::T_WS].id);

        for (size_t r = 0; r < residuals.size(); ++r) {
          if(std::dynamic_pointer_cast<ceres::PoseError>(
            residuals[r].errorInterfacePtr)){ // avoids linearising initial pose error
            mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
            reDoFixation = true;
            continue;
          }
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
              residuals[r].errorInterfacePtr);
          if(!reprojectionError){   // we make sure no reprojection errors are yet included.
            marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
          }
        }

        // add remaining error terms of the sensor states.
        size_t i = SensorStates::Camera;
        for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
          size_t k = CameraSensorStates::T_SCi;
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
            ->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if(checkit->second.sensors[i][j][k].exists &&
             checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
            continue;
          }
          it->second.sensors[i][j][k].exists = false; // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
            it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
            if(!reprojectionError){   // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }

        // now finally we treat all the observations.
        OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0, "bug");
        uint64_t currentKfId = allLinearizedFrames.at(0);

        {
          for(PointMap::iterator pit = landmarksMap_.begin();
              pit != landmarksMap_.end(); ){

            ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

            // first check if we can skip
            bool skipLandmark = true;
            bool hasNewObservations = false;
            bool justDelete = false;
            bool marginalize = true;
            bool errorTermAdded = false;
            std::map<uint64_t,bool> visibleInFrame;
            size_t obsCount = 0;
            for (size_t r = 0; r < residuals.size(); ++r) {
              std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
              if (reprojectionError) {
                uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
                // since we have implemented the linearisation to account for robustification,
                // we don't kick out bad measurements here any more like
                // if(vectorContains(allLinearizedFrames,poseId)){ ...
                //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
                // }
                if(vectorContains(removeFrames,poseId)){
                  skipLandmark = false;
                }
                if(poseId>=currentKfId){
                  marginalize = false;
                  hasNewObservations = true;
                }
                if(vectorContains(allLinearizedFrames, poseId)){
                  visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
                  obsCount++;
                }
              }
            }

            if(residuals.size()==0){
              mapPtr_->removeParameterBlock(pit->first);
              removedLandmarks.push_back(pit->second);
              pit = landmarksMap_.erase(pit);
              continue;
            }

            if(skipLandmark) {
              pit++;
              continue;
            }

            // so, we need to consider it.
            for (size_t r = 0; r < residuals.size(); ++r) {
              std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
              if (reprojectionError) {
                uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
                if((vectorContains(removeFrames,poseId) && hasNewObservations) ||
                   (!vectorContains(allLinearizedFrames,poseId) && marginalize)){
                  // ok, let's ignore the observation.
                  removeObservation(residuals[r].residualBlockId);
                  residuals.erase(residuals.begin() + r);
                  r--;
                } else if(marginalize && vectorContains(allLinearizedFrames,poseId)) {
                  // TODO: consider only the sensible ones for marginalization
                  if(obsCount<2){ //visibleInFrame.size()
                    removeObservation(residuals[r].residualBlockId);
                    residuals.erase(residuals.begin() + r);
                    r--;
                  } else {
                    // add information to be considered in marginalization later.
                    errorTermAdded = true;
                    marginalizationErrorPtr_->addResidualBlock(
                      residuals[r].residualBlockId, false);
                  }
                }
                // check anything left
                if (residuals.size() == 0) {
                  justDelete = true;
                  marginalize = false;
                }
              }
            }

            if(justDelete){
              mapPtr_->removeParameterBlock(pit->first);
              removedLandmarks.push_back(pit->second);
              pit = landmarksMap_.erase(pit);
              continue;
            }
            if(marginalize&&errorTermAdded){
              paremeterBlocksToBeMarginalized.push_back(pit->first);
              keepParameterBlocks.push_back(false);
              removedLandmarks.push_back(pit->second);
              pit = landmarksMap_.erase(pit);
              continue;
            }

            pit++;
          }
        }

        // update book-keeping and go to the next frame
        //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
        if(true) { ///// DEBUG
          multiFramePtrMap_.erase(it->second.id);
          statesMap_.erase(it->second.id);
        }
      }


      return true;

    }

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
    bool Estimator::applyMarginalizationStrategy(
        size_t numKeyframes, size_t numImuFrames,
        okvis::MapPointVector& removedLandmarks)
    {
        // remove linear marginalizationError, if existing
        if (marginalizationErrorPtr_ && marginalizationResidualId_) {
            bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
            OKVIS_ASSERT_TRUE_DBG(Exception, success,
                                  "could not remove marginalization error");
            marginalizationResidualId_ = 0;
            if (!success)
                return false;
        }

        // these will keep track of what we want to marginalize out.
        std::vector<uint64_t> paremeterBlocksToBeMarginalized;
        std::vector<bool> keepParameterBlocks;

        if (!marginalizationErrorPtr_) {
            marginalizationErrorPtr_.reset(
                new ceres::MarginalizationError(*mapPtr_.get()));
        }

        bool reDoFixation = false;
        if(!collectParametersToMargianlize(numKeyframes, numImuFrames,
                                          paremeterBlocksToBeMarginalized, keepParameterBlocks,
                                           removedLandmarks,
                                           reDoFixation)) {
          return true;
        }



        // now apply the actual marginalization
        if(paremeterBlocksToBeMarginalized.size()>0){
            std::vector< ::ceres::ResidualBlockId> addedPriors;
            marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
        }

        // update error computation
        if(paremeterBlocksToBeMarginalized.size()>0){
            marginalizationErrorPtr_->updateErrorComputation();
        }

        // add the marginalization term again
        if(marginalizationErrorPtr_->num_residuals()==0){
            marginalizationErrorPtr_.reset();
        }
        if (marginalizationErrorPtr_) {
            std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
            marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
            marginalizationResidualId_ = mapPtr_->addResidualBlock(
                marginalizationErrorPtr_, NULL, parameterBlockPtrs);
            OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_,
                                  "could not add marginalization error");
            if (!marginalizationResidualId_)
                return false;
        }

        if(reDoFixation){
            // finally fix the first pose properly
            //mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
            okvis::kinematics::Transformation T_WS_0;
            get_T_WS(statesMap_.begin()->first, T_WS_0);
            Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
            information(5,5) = 1.0e14; information(0,0) = 1.0e14; information(1,1) = 1.0e14; information(2,2) = 1.0e14;
            std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
            mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
        }

        return true;
    }




}