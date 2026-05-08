import { useQuery } from '@tanstack/react-query';
import { getPinnedWorksForModel, type WorkTree } from '../fetchers/workspace';

export const usePinnedWorkflows = (modelAlias: string) => {
  return useQuery({
    queryKey: ['pinnedWorkflows', modelAlias],
    queryFn: () => getPinnedWorksForModel(modelAlias),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const generateKernelSliceUrl = (
  modelAlias: string,
  inputAlias: string,
  workAlias: string,
  nodeId: string,
  kernelIndex: string,
  inputIndex: string
): string => {
  return `/models/${modelAlias}/${inputAlias}/${workAlias}/kernel-slice/${nodeId}/${kernelIndex}/${inputIndex}`;
};

export const generatePinnedWorkflowLinks = (
  pinnedWorks: WorkTree[],
  modelAlias: string,
  nodeId: string,
  kernelIndex: string,
  inputIndex: string,
  currentInputAlias: string,
  currentWorkAlias: string
) => {
  return pinnedWorks
    .filter((work) => {
      const [inputAlias, workAlias] = work.alias.split('/');
      // Filter out the current work
      return !(inputAlias === currentInputAlias && workAlias === currentWorkAlias);
    })
    .map((work) => {
      const [inputAlias, workAlias] = work.alias.split('/');
      return {
        url: generateKernelSliceUrl(modelAlias, inputAlias, workAlias, nodeId, kernelIndex, inputIndex),
        name: work.name,
        alias: work.alias,
      };
    });
};