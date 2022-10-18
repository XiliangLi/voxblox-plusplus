// Copyright (c) 2022, lxl, ZJU, China
// Licensed under the BSD 3-Clause License (see LICENSE for details)

// TODO(ntonci): Fix file extension. These files in global_segment_map_node have
// cpp extension and all others have cc.

#include "global_segment_map_node/label_tsdf_server.h"

#include <stdlib.h>

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/TransformStamped.h>
#include <global_segment_map/label_voxel.h>
#include <global_segment_map/utils/file_utils.h>
#include <global_segment_map/utils/map_utils.h>
#include <glog/logging.h>
#include <minkindr_conversions/kindr_tf.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxblox/alignment/icp.h>
#include <voxblox/core/common.h>
#include <voxblox/io/sdf_ply.h>
#include <voxblox_ros/mesh_vis.h>
#include "global_segment_map_node/conversions.h"

#include "voxblox_ros/ros_params.h"

#ifdef APPROXMVBB_AVAILABLE
#include <ApproxMVBB/ComputeApproxMVBB.hpp>
#endif

namespace voxblox {
namespace voxblox_lts {

LabelTsdfServer::LabelTsdfServer(const ros::NodeHandle& nh,
                                 const ros::NodeHandle& nh_private)
    : LabelTsdfServer(nh, nh_private, getLabelTsdfMapConfigFromRosParam(nh_private),
                        getTsdfIntegratorConfigFromRosParam(nh_private),
                        getMeshIntegratorConfigFromRosParam(nh_private)) {}

LabelTsdfServer::LabelTsdfServer(const ros::NodeHandle& nh, 
                                 const ros::NodeHandle& nh_private,
                                 const LabelTsdfMap::Config& config,
                                 const LabelTsdfIntegrator::Config& integrator_config,
                                 const MeshIntegratorConfig& mesh_config)
    : nh_(nh),
      nh_private_(nh_private),
      tf_listener_(ros::Duration(500)),
      world_frame_("world"),
      map_config_(config),
      tsdf_integrator_config_(integrator_config),
      mesh_config_(mesh_config) {
    getServerConfigFromRosParam(nh_private_);

    if(verbose_log_) {
        FLAGS_stderrthreshold = 0;
    }

    map_.reset(new LabelTsdfMap(config));

    integrator_.reset(new LabelTsdfIntegrator(
        tsdf_integrator_config_, label_tsdf_integrator_config_, map_.get()));

    // Visualization settings
    bool visualize = false;
    nh_private_.param<bool>("meshing/visualize", visualize, visualize);

    bool save_visualizer_frames = false;
    nh_private_.param<bool>("debug/save_visualizer_frames",
                            save_visualizer_frames,
                            save_visualizer_frames);
    bool multiple_visualizers = false;
    nh_private_.param<bool>("debug/multiple_visualizers",
                            multiple_visualizers_,
                            multiple_visualizers_);

    mesh_merged_layer_.reset(new MeshLayer(map_->block_size()));

    if(multiple_visualizers_) {
        mesh_label_layer_.reset(new MeshLayer(map_->block_size()));
        mesh_semantic_layer_.reset(new MeshLayer(map_->block_size()));
        mesh_instance_layer_.reset(new MeshLayer(map_->block_size()));
    }

    resetMeshIntegrators();

    std::vector<double> camera_position;
    std::vector<double> clip_distances;
    nh_private_.param<std::vector<double>>(
        "meshing/visualizer_parameters/camera_position",
        camera_position, 
        camera_position);
    nh_private_.param<std::vector<double>>(
        "meshing/visualizer_parameters/clip_distances",
        clip_distances,
        clip_distances);
    if(visualize) {
        std::vector<std::shared_ptr<MeshLayer>> mesh_layers;
        mesh_layers.push_back(mesh_merged_layer_);
        if(multiple_visualizers_) {
            mesh_layers.push_back(mesh_label_layer_);
            mesh_layers.push_back(mesh_instance_layer_);
            mesh_layers.push_back(mesh_semantic_layer_);
        }
        visualizer_ = new Visualizer(mesh_layers, &mesh_layer_updated_, &mesh_layer_mutex, 
                                        camera_position, clip_distances, save_visualizer_frames);
        viz_thread_ = std::thread(&Visualizer::visualizeMesh, visualizer_);

    }

    // If set, use a timer to progressively update the mesh.
    double update_mesh_every_n_sec = 0.0;
    nh_private_.param<double>("meshing/update_mesh_every_n_sec",
                                      update_mesh_every_n_sec,
                                      update_mesh_every_n_sec);

    if (update_mesh_every_n_sec > 0.0) {
      update_mesh_timer_ = nh_private_.createTimer(
          ros::Duration(update_mesh_every_n_sec), &LabelTsdfServer::updateMeshEvent,
          this);
    }

    nh_private_.param<std::string>("meshing/mesh_filename",
                                    mesh_filename_, mesh_filename_);

    
    

}

void LabelTsdfServer::getServerConfigFromRosParam(const ros::NodeHandle& nh_private) {
    // Determine label integrator parameters
    nh_private.param<bool>(
      "pairwise_confidence_merging/enable_pairwise_confidence_merging",
      label_tsdf_integrator_config_.enable_pairwise_confidence_merging,
      label_tsdf_integrator_config_.enable_pairwise_confidence_merging);
    nh_private.param<FloatingPoint>(
      "pairwise_confidence_merging/merging_min_overlap_ratio",
      label_tsdf_integrator_config_.merging_min_overlap_ratio,
      label_tsdf_integrator_config_.merging_min_overlap_ratio);
    nh_private.param<int>(
      "pairwise_confidence_merging/merging_min_frame_count",
      label_tsdf_integrator_config_.merging_min_frame_count,
      label_tsdf_integrator_config_.merging_min_frame_count);

    nh_private.param<bool>(
      "semantic_instance_segmentation/enable_semantic_instance_segmentation",
      label_tsdf_integrator_config_.enable_semantic_instance_segmentation,
      label_tsdf_integrator_config_.enable_semantic_instance_segmentation);

    enable_semantic_instance_segmentation_ =
      label_tsdf_integrator_config_.enable_semantic_instance_segmentation;

    std::string class_task("coco80");
    nh_private.param<std::string>(
      "semantic_instance_segmentation/class_task", class_task, class_task);
    if (class_task.compare("coco80") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCoco80;
    } else if (class_task.compare("nyu13") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kNyu13;
    } else {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask::kCoco80;
    }

    nh_private.param<bool>("icp/enable_icp",
                            label_tsdf_integrator_config_.enable_icp,
                            label_tsdf_integrator_config_.enable_icp);
    nh_private.param<bool>("icp/keep_track_of_icp_correction",
                            label_tsdf_integrator_config_.keep_track_of_icp_correction,
                            label_tsdf_integrator_config_.keep_track_of_icp_correction);
    
    // Determine class member variable
    nh_private.param<bool>("debug/verbose_log",
                             verbose_log_,
                             verbose_log_);
    nh_private.param<std::string>("world_frame_id",
                                     world_frame_, 
                                     world_frame_);
    nh_private.param<bool>("publishers/publish_scene_map",
                            publish_scene_map_,
                            publish_scene_map_);
    nh_private.param<bool>("publishers/publish_object_bbox",
                            publish_object_bbox_,
                            publish_object_bbox_);
    nh_private.param<bool>("use_label_propagation",
                            use_label_propagation_,
                            use_label_propagation_);                                                                                             


}
  
} // namespace voxblox_lts
} // namespace voxblox


