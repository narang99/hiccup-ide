#!/usr/bin/env vite-node

// Import utilities directly from existing frontend modules
import { startPruning, saveWorkSaliencyMaps, finalizePruning, getPruningStatus } from '../src/fetchers/graph.js';
import { getTopKThreshold } from '../src/utils/topk.js';
import { loadLayerSaliencyMaps } from '../src/fetchers/saliency_map.js';
import type { CoordinateAlgorithm, PruningStatusResponse } from '../src/fetchers/graph.js';
import type { LayerSaliencyData } from '../src/fetchers/saliency_map.js';

// Layer processing order (output → input)
const LAYER_ORDER = ["layers.3", "layers.2", "layers.1", "layers.0", "x"];

interface AutoPruneOptions {
  modelAlias: string;
  inputAlias: string; 
  workAlias: string;
  thresholdPercent: number; // e.g., 85 for 85% retention
}

/**
 * Automatically prune a neural network model keeping the top percentage of contributions
 */
async function autoPrune(options: AutoPruneOptions): Promise<void> {
  const { modelAlias, inputAlias, workAlias, thresholdPercent } = options;
  
  console.log(`🚀 Starting automated pruning for ${modelAlias}/${inputAlias}/${workAlias}`);
  console.log(`📊 Keeping top ${thresholdPercent}% of contributions`);
  
  try {
    // 1. Initialize pruning session
    console.log('🔄 Starting pruning session...');
    const startResult = await startPruning(modelAlias, inputAlias, workAlias);
    console.log(`✅ Session started - cloned ${startResult.cloned_count} saliency maps to scratchpad`);
    
    // 2. Check initial status
    let status = await getPruningStatus(modelAlias, inputAlias, workAlias);
    console.log(`📋 Found ${status.layers.total.length} layers to process: ${status.layers.total.join(', ')}`);
    
    // 3. Process each layer sequentially
    for (const layerName of LAYER_ORDER) {
      // Check if this layer is already done
      if (status.layers.done.includes(layerName)) {
        console.log(`⏭️  Layer ${layerName} already processed, skipping...`);
        continue;
      }
      
      console.log(`\n🎯 Processing layer: ${layerName}`);
      
      // Fetch all saliency data for this layer
      console.log('  📥 Fetching saliency data...');
      const saliencyData = await loadLayerSaliencyMaps(modelAlias, inputAlias, layerName, workAlias, status.session_active);
      // console.log("muhahaha", saliencyData.items.filter(d => d.coordinate_type === "output_channel"))
      
      // Filter for current layer's output_channel coordinates
      const layerCoordinates = saliencyData.items.filter(item => 
        item.layer_name === layerName && item.coordinate_type === 'output_channel'
      );
      
      if (layerCoordinates.length === 0) {
        console.log(`  ⚠️  No coordinates found for layer ${layerName}, skipping...`);
        continue;
      }
      
      console.log(`  🔍 Found ${layerCoordinates.length} coordinates for layer ${layerName}`);
      
      // console.log("layer coordinatessss", layerCoordinates);
      // Collect all values for threshold calculation
      const allValues: number[] = [];
      for (const coord of layerCoordinates) {
        const data = coord.data;
        if (Array.isArray(data)) {
          // Handle 2D array (flatten)
          for (const row of data) {
            if (Array.isArray(row)) {
              allValues.push(...row);
            } else {
              allValues.push(row);
            }
          }
        } else {
          allValues.push(data);
        }
      }
      
      // Calculate threshold for desired retention percentage
      const threshold = getTopKThreshold(allValues, thresholdPercent / 100);
      console.log(`  🧮 Calculated threshold: ${threshold} (keeping top ${thresholdPercent}% of ${allValues.length} values)`);
      
      // Prepare coordinate algorithms for API call
      const coordinateAlgorithms: CoordinateAlgorithm[] = layerCoordinates.map(coord => ({
        coordinate: coord.coordinate,
        algorithm: {
          type: 'ThresholdAlgorithm',
          threshold: threshold
        }
      }));
      
      // Apply pruning to current layer
      console.log(`  🔧 Applying threshold algorithm to ${coordinateAlgorithms.length} coordinates...`);
      const pruneResult = await saveWorkSaliencyMaps(
        modelAlias, 
        inputAlias, 
        workAlias, 
        coordinateAlgorithms
      );
      
      console.log(`  ✅ Layer ${layerName} processed - updated ${pruneResult.updated} coordinates`);
      
      // Check status to confirm layer completion
      status = await getPruningStatus(modelAlias, inputAlias, workAlias);
      
      // Small delay to avoid overwhelming the backend
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // 4. Finalize pruning session
    console.log('\n🏁 Finalizing pruning session...');
    const finalizeResult = await finalizePruning(modelAlias, inputAlias, workAlias);
    console.log(`✅ Finalization complete - committed ${finalizeResult.committed_count} modified maps`);
    
    // 5. Final status check
    const finalStatus = await getPruningStatus(modelAlias, inputAlias, workAlias);
    console.log(`\n📊 Final status:`);
    console.log(`   ✓ Completed layers: ${finalStatus.layers.done.length}/${finalStatus.layers.total.length}`);
    console.log(`   ✓ Session active: ${finalStatus.session_active}`);
    
    console.log(`\n🎉 Automated pruning completed successfully!`);
    
  } catch (error) {
    console.error(`❌ Pruning failed:`, error);
    throw error;
  }
}

/**
 * Command-line interface
 */
async function main() {
  const args = process.argv.slice(2);
  console.log("argsssss", args)
  
  if (args.length < 4) {
    console.log(`
Usage: npm run auto-prune <modelAlias> <inputAlias> <workAlias> <thresholdPercent>

Examples:
  npm run auto-prune simple-mnist sample1 automated-85 85
  npm run auto-prune simple-mnist sample2 automated-90 90

Parameters:
  modelAlias       - Model identifier (e.g., 'simple-mnist')
  inputAlias       - Input identifier (e.g., 'sample1') 
  workAlias        - Work/workflow name (e.g., 'automated-85')
  thresholdPercent - Percentage of contributions to keep (e.g., 85)
`);
    process.exit(1);
  }
  
  const [modelAlias, inputAlias, workAlias, thresholdPercentStr] = args;
  const thresholdPercent = parseFloat(thresholdPercentStr);
  
  if (isNaN(thresholdPercent) || thresholdPercent <= 0 || thresholdPercent > 100) {
    console.error('❌ thresholdPercent must be a number between 0 and 100');
    process.exit(1);
  }
  
  console.log(`start pruning: modelAlias=${modelAlias} inputAlais=${inputAlias} workAlais=${workAlias} threshold=${thresholdPercent}`)
  await autoPrune({
    modelAlias,
    inputAlias, 
    workAlias,
    thresholdPercent
  });
}


// Export for potential use as a module
export { autoPrune, type AutoPruneOptions };

// // Run if called directly
// if (require.main === module) {
//   console.log("aaaaaaaaaa")
// }
main().catch(error => {
  console.error('❌ Script failed:', error);
  process.exit(1);
});