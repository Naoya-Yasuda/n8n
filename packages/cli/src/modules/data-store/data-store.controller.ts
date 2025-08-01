import type {
	AddDataStoreColumnDto,
	CreateDataStoreDto,
	DeleteDataStoreColumnDto,
	ListDataStoreContentQueryDto,
	ListDataStoreQueryDto,
	MoveDataStoreColumnDto,
	UpdateDataStoreDto,
	DataStoreRows,
} from '@n8n/api-types';
import { AuthenticatedRequest } from '@n8n/db';
import {
	Body,
	Delete,
	Get,
	Param,
	Patch,
	Post,
	ProjectScope,
	RestController,
} from '@n8n/decorators';

import { DataStoreService } from './data-store.service';

@RestController('/projects/:projectId/data-stores')
export class DataStoreController {
	constructor(private readonly dataStoreService: DataStoreService) {}

	@Post('/')
	@ProjectScope('dataStore:create')
	async createDataStore(
		req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Body dto: CreateDataStoreDto,
	) {
		return await this.dataStoreService.createDataStore(req.params.projectId, dto);
	}

	@Get('/')
	@ProjectScope('dataStore:list')
	async listDataStores(
		req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Body payload: Partial<ListDataStoreQueryDto> = {},
	) {
		const providedFilter = payload?.filter ?? {};
		const [data, count] = await this.dataStoreService.getManyAndCount({
			...payload,
			filter: { ...providedFilter, projectId: req.params.projectId },
		});

		return { count, data };
	}

	@Patch('/:dataStoreId')
	@ProjectScope('dataStore:update')
	async updateDataStore(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
		@Body dto: UpdateDataStoreDto,
	) {
		return await this.dataStoreService.updateDataStore(dataStoreId, dto);
	}

	@Delete('/:dataStoreId')
	@ProjectScope('dataStore:delete')
	async deleteDataStore(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
	) {
		return await this.dataStoreService.deleteDataStore(dataStoreId);
	}

	@Get('/:dataStoreId/columns')
	@ProjectScope('dataStore:read')
	async getColumns(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
	) {
		return await this.dataStoreService.getColumns(dataStoreId);
	}

	@Post('/:dataStoreId/columns')
	@ProjectScope('dataStore:update')
	async addColumn(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
		@Body dto: AddDataStoreColumnDto,
	) {
		return await this.dataStoreService.addColumn(dataStoreId, dto);
	}

	@Delete('/:dataStoreId/columns')
	@ProjectScope('dataStore:update')
	async deleteColumn(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
		@Body dto: DeleteDataStoreColumnDto,
	) {
		return await this.dataStoreService.deleteColumn(dataStoreId, dto);
	}

	@Patch('/:dataStoreId/columns/:columnId/move')
	@ProjectScope('dataStore:update')
	async moveColumn(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
		@Param('columnId') columnId: string,
		@Body dto: MoveDataStoreColumnDto,
	) {
		return await this.dataStoreService.moveColumn(dataStoreId, columnId, dto);
	}

	@Get('/:dataStoreId/rows')
	@ProjectScope('dataStore:readRow')
	async getDataStoreRows(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
		@Body dto: Partial<ListDataStoreContentQueryDto>,
	) {
		return await this.dataStoreService.getManyRowsAndCount(dataStoreId, dto);
	}

	@Post('/:dataStoreId/append')
	@ProjectScope('dataStore:writeRow')
	async appendDataStoreRows(
		_req: AuthenticatedRequest<{ projectId: string }>,
		_res: Response,
		@Param('dataStoreId') dataStoreId: string,
		@Body dto: DataStoreRows,
	) {
		return await this.dataStoreService.appendRows(dataStoreId, dto);
	}
}
