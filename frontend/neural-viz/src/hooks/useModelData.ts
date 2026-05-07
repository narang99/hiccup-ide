import { useEffect } from 'react';
import { useModelDataCacheStore } from '../stores/modelDataCacheStore';
import { type ModelData } from '../types/model';

interface UseModelDataResult {
  modelData: ModelData | null;
  loading: boolean;
  error: string | null;
}

/*
Example data format that is provided by useModelData (this is what the backend returns)
{
  "nodes": [
    {
      "id": "x",
      "type": "Input",
      "params": {},
      "shape": [1, 1, 28, 28]
    },
    {
      "id": "layers.0",
      "type": "Conv2d",
      "params": {
        "in_channels": 1,
        "out_channels": 8,
        "kernel_size": [3, 3],
        "stride": [2, 2],
        "padding": [1, 1]
      },
      "shape": [1, 8, 14, 14]
    },
    {
      "id": "layers.1",
      "type": "ReLU",
      "params": {},
      "shape": [ 1, 8, 14, 14 ]
    },
    {
      "id": "layers.2",
      "type": "Conv2d",
      "params": {
        "in_channels": 8,
        "out_channels": 16,
        "kernel_size": [ 3, 3 ],
        "stride": [ 2, 2 ],
        "padding": [ 1, 1 ]
      },
      "shape": [ 1, 16, 7, 7 ]
    },
    {
      "id": "layers.3",
      "type": "ReLU",
      "params": {},
      "shape": [ 1, 16, 7, 7 ]
    },
    {
      "id": "layers.4",
      "type": "Flatten",
      "params": {},
      "shape": [ 1, 784 ]
    },
    {
      "id": "layers.5",
      "type": "Linear",
      "params": {},
      "shape": [ 1, 10 ]
    },
    {
      "id": "output",
      "type": "Output",
      "params": {},
      "shape": []
    }
  ],
  "edges": [
    {
      "source": "x",
      "target": "layers.0"
    },
    {
      "source": "layers.0",
      "target": "layers.1"
    },
    {
      "source": "layers.1",
      "target": "layers.2"
    },
    {
      "source": "layers.2",
      "target": "layers.3"
    },
    {
      "source": "layers.3",
      "target": "layers.4"
    },
    {
      "source": "layers.4",
      "target": "layers.5"
    },
    {
      "source": "layers.5",
      "target": "output"
    }
  ]
}
*/

export const useModelData = (modelAlias: string, apiBaseUrl?: string): UseModelDataResult => {
  const { fetchModelData, getModelData } = useModelDataCacheStore();

  const cacheEntry = getModelData(modelAlias);

  useEffect(() => {
    fetchModelData(modelAlias, apiBaseUrl);
  }, [modelAlias, apiBaseUrl, fetchModelData]);

  return {
    modelData: cacheEntry?.status === 'success' ? cacheEntry.data : null,
    loading: cacheEntry?.status === 'loading' || !cacheEntry,
    error: cacheEntry?.status === 'error' ? cacheEntry.error : null,
  };
};