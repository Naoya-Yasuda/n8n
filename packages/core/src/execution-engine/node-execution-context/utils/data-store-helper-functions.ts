import type {
	DataStoreProxyFunctions,
	INode,
	Workflow,
	IWorkflowExecuteAdditionalData,
} from 'n8n-workflow';

export function getDataStoreHelperFunctions(
	additionalData: IWorkflowExecuteAdditionalData,
	workflow: Workflow,
	node: INode,
): DataStoreProxyFunctions {
	return {
		getProxy: async <T extends string | undefined>(dataStoreId?: T) =>
			await additionalData.dataStoreProxy.getProxy(workflow, node, dataStoreId),
	};
}
