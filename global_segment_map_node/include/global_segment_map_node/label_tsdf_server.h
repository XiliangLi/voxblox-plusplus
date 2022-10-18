#ifndef GSM_LABEL_TSDF_SERVER_H_
#define GSM_LABEL_TSDF_SERVER_H_

// gsm
#include <vector>

#include <geometry_msgs/Transform.h>
#include <global_segment_map/label_tsdf_integrator.h>
#include <global_segment_map/label_tsdf_map.h>
#include <global_segment_map/label_voxel.h>
#include <global_segment_map/meshing/label_tsdf_mesh_integrator.h>
#include <global_segment_map/utils/visualizer.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox_ros/conversions.h>
#include <vpp_msgs/GetAlignedInstanceBoundingBox.h>
#include <vpp_msgs/GetListSemanticInstances.h>
#include <vpp_msgs/GetMap.h>
#include <vpp_msgs/GetScenePointcloud.h>

// voxblox

/*
#include <deque>
#include <memory>
#include <queue>
#include <string>

#include <minkindr_conversions/kindr_msg.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include <voxblox/alignment/icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/merge_integration.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/FilePath.h>
#include <voxblox_msgs/Mesh.h>

#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"
*/

namespace voxblox {
namespace voxblox_lts {

constexpr float kDefaultMaxIntensity = 100.0;

class LabelTsdfServer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LabelTsdfServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    LabelTsdfServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private, 
                    const LabelTsdfMap::Config& config,
                    const LabelTsdfIntegrator::Config& integrator_config,
                    const MeshIntegratorConfig& mesh_config);
    virtual ~LabelTsdfServer() {}

    void getServerConfigFromRosParam(const ros::NodeHandle& nh_private);

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

protected:

    void updateMeshEvent(const ros::TimerEvent& e);

    // Not Thread Safe
    void resetMeshIntegrators();

    std::shared_ptr<LabelTsdfMap> map_;
    std::shared_ptr<LabelTsdfIntegrator> integrator_;

    std::shared_ptr<MeshLayer> mesh_label_layer_;
    std::shared_ptr<MeshLayer> mesh_semantic_layer_;
    std::shared_ptr<MeshLayer> mesh_instance_layer_;
    std::shared_ptr<MeshLayer> mesh_merged_layer_;


    LabelTsdfMap::Config map_config_;
    LabelTsdfIntegrator::Config tsdf_integrator_config_;
    LabelTsdfIntegrator::LabelTsdfConfig label_tsdf_integrator_config_;

    MeshIntegratorConfig mesh_config_;
    MeshLabelIntegrator::LabelTsdfConfig label_tsdf_mesh_config_;
    ros::Timer update_mesh_timer_;

    bool verbose_log_ = false;

    size_t integrated_frames_count_ = 0u;
    std::string world_frame_;
    std::string mesh_filename_;
    
    bool integration_on_ = true;
    bool publish_scene_map_ = false;
    bool publish_scene_mesh_ = false;
    bool publish_object_bbox_ = false;
    bool mesh_layer_updated_ = false;
    bool use_label_propagation_ = true;
    bool enable_semantic_instance_segmentation_ = true;

    bool received_first_message = false;
    bool need_full_remesh_ = false;

    tf::TransformListener tf_listener_;
    
    // visualize
    bool multiple_visualizers_;
    bool mesh_layer_updated_;
    bool need_full_remesh_;
    std::thread viz_thread_;
    Visualizer* visualizer_;
    std::mutex label_tsdf_layers_mutex;
    std::mutex mesh_layer_mutex;

    
};
} // namespace voxbloc_lts
} // namespace voxblox



#endif // GSM_LABEL_TSDF_SERVER_H_