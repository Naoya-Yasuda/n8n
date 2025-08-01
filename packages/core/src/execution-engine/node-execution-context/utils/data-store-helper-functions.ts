import type { IDataStoreService } from '@n8n/api-types';
import type {
	DataStoreProxyFunctions,
	INode,
	Workflow,
	IDataStoreProjectService,
} from 'n8n-workflow';

export function getDataStoreHelperFunctions(
	dataStoreService: IDataStoreService,
	workflow: Workflow,
	node: INode,
): DataStoreProxyFunctions {
	return {
		dataStoreProxy: (): IDataStoreProjectService => {
			console.log(node.type);
			// if (node.type !== 'n8n-nodes-base.dataStore') {
			// 	throw new Error('This helper is only available for data store nodes');
			// }

			return dataStoreService as never;
		},
	};
}
