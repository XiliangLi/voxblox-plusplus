#ifndef GLOBAL_SEGMENT_MAP_UTILS_MAP_UTILS_H_
#define GLOBAL_SEGMENT_MAP_UTILS_MAP_UTILS_H_

#include <global_segment_map/common.h>
#include <global_segment_map/label_tsdf_map.h>
#include <global_segment_map/semantic_instance_label_fusion.h>

namespace voxblox {

void serializeMapAsMsg(std::shared_ptr<const LabelTsdfMap> map,
                        const bool only_updated,
                        voxblox_msgs::Layer* msg,
                        const MapDerializationAction& action = MapDerializationAction::kUpdate) {
  CHECK_NOTNULL(msg);
  CHECK_NOTNULL(msg);
  const Layer<TsdfVoxel>& tsdf_layer = map->getTsdfLayer();
  const Layer<LabelVoxel>& label_layer = map->getLabelLayer();

  const SemanticInstanceLabelFusion semantic_instance_label_fusion = 
      map->getSemanticInstanceLabelFusion();


  msg->voxel_size = tsdf_layer.voxel_size();
  msg->voxels_per_side = tsdf_layer.voxels_per_side();
  msg->layer_type = voxel_types::kTsdf;

  BlockIndexList block_list;
  if(only_updated) {
    tsdf_layer.getAllUpdatedBlocks(&block_list);
  } else {
    tsdf_layer.getAllAllocatedBlocks(&block_list);
  }

  msg->action = static_cast<uint8_t>(action);

  voxblox_msgs::Block block_msg;
  msg->blocks.reserve(block_list.size());

  // block paramers
  constexpr size_t kNumDataPacketsPerVoxel = 5u;
  size_t vps = tsdf_layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Iterate over all blocks
  for(const BlockIndex& index : block_list) {
    block_msg.x_index = index.x();
    block_msg.y_index = index.y();
    block_msg.z_index = index.z();

    std::vector<uint32_t> data;
    
    data.clear();
    data.reserve(num_voxels_per_block * kNumDataPacketsPerVoxel);
    // tsdf_layer.getBlockByIndex(index).serializeToIntegers(&data);

    const Block<TsdfVoxel>& tsdf_block = tsdf_layer.getBlockByIndex(index);
    const Block<LabelVoxel>& label_block = label_layer.getBlockByIndex(index);

    //Iterate over all voxels  in said blocks
    for(size_t linear_index = 0; linear_index < num_voxels_per_block; 
        linear_index++) {
      TsdfVoxel tsdf_voxel = tsdf_block.getVoxelByLinearIndex(linear_index);
      LabelVoxel label_voxel = label_block.getVoxelByLinearIndex(linear_index);

      constexpr float kMinWeight = 0.0f;
      constexpr float kFramesCountThresholdFactor = 0.1f;

      // if(tsdf_voxel.weight > kMinWeight) {
        // voxel.weight voxel.distance
        const uint32_t* bytes_1_ptr = 
            reinterpret_cast<const uint32_t*>(&tsdf_voxel.distance);
        data.push_back(*bytes_1_ptr);

        const uint32_t* bytes_2_ptr = 
            reinterpret_cast<const uint32_t*>(&tsdf_voxel.weight);
        data.push_back(*bytes_2_ptr);

        // get voxel color
        Label segment_label = label_voxel.label;
        SemanticLabel semantic_class = BackgroundLabel;
        SemanticPair semantic_count_pair;
        InstanceLabel instance_id = 
            semantic_instance_label_fusion.getInstanceLabel(
                segment_label, kFramesCountThresholdFactor);
        
        if(instance_id) {
          semantic_count_pair = semantic_instance_label_fusion.getSemanticLabel(
              label_voxel.label);
          semantic_class = semantic_count_pair.semantic_label;  
        }

        tsdf_voxel.semantic_label = semantic_class;
        tsdf_voxel.semantic_count = semantic_count_pair.semantic_count;
        
        // add semantic information
        const uint32_t* bytes_3_ptr = 
            reinterpret_cast<const uint32_t*>(&tsdf_voxel.semantic_label);
        data.push_back(*bytes_3_ptr);

        const uint32_t* bytes_4_ptr = 
            reinterpret_cast<const uint32_t*>(&tsdf_voxel.semantic_count);
        data.push_back(*bytes_4_ptr);

        voxblox::Color color;
        map->semantic_color_map_.getColor(semantic_class, &color);
        color.a = static_cast<uint8_t>(255);
        tsdf_voxel.color = color;

        data.push_back(static_cast<uint32_t>(tsdf_voxel.color.a) |
                (static_cast<uint32_t>(tsdf_voxel.color.b) << 8) |
                (static_cast<uint32_t>(tsdf_voxel.color.g) << 16) |
                (static_cast<uint32_t>(tsdf_voxel.color.r) << 24));
        
      // }
    }

    block_msg.data = data;
    msg->blocks.push_back(block_msg);
  }

}

inline void createPointcloudFromMap(const LabelTsdfMap& map,
                                    pcl::PointCloud<PointMapType>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  pointcloud->clear();

  const Layer<TsdfVoxel>& tsdf_layer = map.getTsdfLayer();
  const Layer<LabelVoxel>& label_layer = map.getLabelLayer();

  const SemanticInstanceLabelFusion semantic_instance_label_fusion =
      map.getSemanticInstanceLabelFusion();

  BlockIndexList blocks;
  tsdf_layer.getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = tsdf_layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  double intensity = 0.0;
  // Iterate over all blocks.
  for (const BlockIndex& index : blocks) {
    const Block<TsdfVoxel>& tsdf_block = tsdf_layer.getBlockByIndex(index);
    const Block<LabelVoxel>& label_block = label_layer.getBlockByIndex(index);

    // Iterate over all voxels in said blocks.
    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = tsdf_block.computeCoordinatesFromLinearIndex(linear_index);

      TsdfVoxel tsdf_voxel = tsdf_block.getVoxelByLinearIndex(linear_index);
      LabelVoxel label_voxel = label_block.getVoxelByLinearIndex(linear_index);

      constexpr float kMinWeight = 0.0f;
      constexpr float kFramesCountThresholdFactor = 0.1f;

      if (tsdf_voxel.weight > kMinWeight) {
        Label segment_label = label_voxel.label;
        SemanticLabel semantic_class = BackgroundLabel;
        SemanticPair semantic_count_pair;
        InstanceLabel instance_id =
            semantic_instance_label_fusion.getInstanceLabel(
                segment_label, kFramesCountThresholdFactor);

        if (instance_id) {
          semantic_count_pair = semantic_instance_label_fusion.getSemanticLabel(
              label_voxel.label);
          semantic_class = semantic_count_pair.semantic_label;
        }

        PointMapType point;

        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();

        point.distance = tsdf_voxel.distance;
        point.weight = tsdf_voxel.weight;

        point.segment_label = segment_label;
        point.semantic_class = semantic_class;
        pointcloud->push_back(point);
      }
    }
  }
}

}  // namespace voxblox

#endif  // GLOBAL_SEGMENT_MAP_UTILS_MAP_UTILS_H_
