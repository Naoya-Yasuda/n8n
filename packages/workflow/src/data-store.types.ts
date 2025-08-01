export type DataStoreColumnType = 'string' | 'number' | 'boolean' | 'date';

export type DataStoreColumn = {
	id: string;
	name: string;
	type: DataStoreColumnType;
	columnIndex: number;
	dataStoreId: string;
};

export type DataStore = {
	id: string;
	name: string;
	columns: DataStoreColumn[];
	createdAt: Date;
	updatedAt: Date;
	projectId: string;
	sizeBytes: number;
};

export type CreateDataStoreColumnPayload = Pick<DataStoreColumn, 'name' | 'type'> &
	Partial<Pick<DataStoreColumn, 'columnIndex'>>;

export type CreateDataStorePayload = Pick<DataStore, 'name'> & {
	columns: CreateDataStoreColumnPayload;
};

export type UpdateDataStorePayload = { name: string };

export type ListDataStoreOptions = {
	filter?: Record<string, string | string[]>;
	sortBy?:
		| 'name:asc'
		| 'name:desc'
		| 'createdAt:asc'
		| 'createdAt:desc'
		| 'updatedAt:asc'
		| 'updatedAt:desc'
		| 'sizeBytes:asc'
		| 'sizeBytes:desc';
	take?: number;
	skip?: number;
};

export type ListDataStoreRowsOptions = {
	filter?: Record<string, string | string[]>;
	sortBy?: Array<[string, 'ASC' | 'DESC']>;
	take?: number;
	skip?: number;
};

export type MoveDataStoreColumn = {
	targetIndex: number;
};

export type DeleteDataStoreColumn = {
	columnId: string;
};

export type DataStoreColumnJsType = string | number | boolean | Date;

export type DataStoreRows = Array<Record<PropertyKey, DataStoreColumnJsType | null>>;

// API for a data store service operating on a specific projectId
export interface IDataStoreProjectService {
	createDataStore(dto: CreateDataStorePayload): Promise<DataStore>;
	updateDataStore(dataStoreId: string, dto: UpdateDataStorePayload): Promise<boolean>;
	getManyAndCount(options: ListDataStoreOptions): Promise<{ count: number; data: DataStore[] }>;

	deleteDataStoreAll(): Promise<boolean>;
	deleteDataStore(dataStoreId: string): Promise<boolean>;

	getColumns(dataStoreId: string): Promise<DataStoreColumn[]>;
	addColumn(dataStoreId: string, dto: CreateDataStoreColumnPayload): Promise<DataStoreColumn>;
	moveColumn(dataStoreId: string, columnId: string, dto: MoveDataStoreColumn): Promise<boolean>;
	deleteColumn(dataStoreId: string, dto: DeleteDataStoreColumn): Promise<boolean>;

	getManyRowsAndCount(
		dataStoreId: string,
		dto: Partial<ListDataStoreRowsOptions>,
	): Promise<{ count: number; data: DataStoreRows[] }>;
	appendRows(dataStoreId: string, rows: DataStoreRows): Promise<boolean>;
}
