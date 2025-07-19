"""Database adapter for MCP-RAG-Control system."""

import re
from typing import Any, Dict, List, Optional, Union

from .base_adapter import BaseAdapter, OperationError
from ..utils.logger import get_logger, log_error_with_context


class DatabaseAdapter(BaseAdapter):
    """Adapter for database operations via MCP."""
    
    def __init__(self, *args, **kwargs):
        """Initialize database adapter with additional security measures."""
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"{__name__}.DatabaseAdapter")
        
        # SQL injection prevention patterns
        self._dangerous_patterns = [
            r'(\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE)\b)',
            r'(--|/\*|\*/)',
            r'(\bUNION\b.*\bSELECT\b)',
            r'(\bEXEC(UTE)?\b)',
            r'(\bxp_\w+)',
            r'(\bsp_\w+)'
        ]
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self._dangerous_patterns]
    
    def _validate_identifier(self, identifier: str) -> bool:
        """Validate SQL identifier (table name, column name, etc.).
        
        Args:
            identifier: SQL identifier to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not identifier or not isinstance(identifier, str):
            return False
        
        # Check length
        if len(identifier) > 64:  # Standard SQL identifier limit
            return False
        
        # Check for valid identifier pattern (alphanumeric + underscore, starting with letter)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', identifier):
            return False
        
        # Check for dangerous patterns
        for pattern in self._compiled_patterns:
            if pattern.search(identifier):
                self.logger.warning(
                    f"Potentially dangerous identifier detected: {identifier}",
                    extra={"identifier": identifier, "module_id": str(self.module.id)}
                )
                return False
        
        return True
    
    def _validate_sql_value(self, value: Any) -> bool:
        """Validate SQL value for potential injection attempts.
        
        Args:
            value: Value to validate
            
        Returns:
            True if safe, False if potentially dangerous
        """
        if value is None or isinstance(value, (int, float, bool)):
            return True
        
        if isinstance(value, str):
            # Check for dangerous patterns in string values
            for pattern in self._compiled_patterns:
                if pattern.search(value):
                    self.logger.warning(
                        f"Potentially dangerous SQL value detected",
                        extra={"value_type": type(value).__name__, "module_id": str(self.module.id)}
                    )
                    return False
        
        return True
    
    async def _get_health_details(self) -> Dict[str, Any]:
        """Get database specific health details."""
        try:
            # Get database info
            response = await self.send_request("database.info", {
                "database_name": self.module.config.database_name
            })
            
            return {
                "database_name": self.module.config.database_name,
                "database_type": response.result.get("database_type"),
                "version": response.result.get("version"),
                "total_tables": response.result.get("total_tables", 0),
                "total_records": response.result.get("total_records", 0),
                "database_size_mb": response.result.get("database_size_mb", 0)
            }
        except Exception:
            return {"database_name": self.module.config.database_name}
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database operation."""
        if operation not in self.get_supported_operations():
            raise OperationError(f"Unsupported operation: {operation}")
        
        method_map = {
            "query": self._execute_query,
            "insert": self._insert_records,
            "update": self._update_records,
            "delete": self._delete_records,
            "create_table": self._create_table,
            "drop_table": self._drop_table,
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "execute_transaction": self._execute_transaction,
            "bulk_insert": self._bulk_insert,
            "export_data": self._export_data,
            "import_data": self._import_data
        }
        
        handler = method_map.get(operation)
        if not handler:
            raise OperationError(f"Operation handler not found: {operation}")
        
        return await handler(parameters)
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get database adapter schema."""
        return {
            "type": "database",
            "operations": {
                "query": {
                    "description": "Execute SQL query",
                    "parameters": {
                        "sql": {"type": "string", "description": "SQL query to execute"},
                        "parameters": {"type": "array", "optional": True},
                        "limit": {"type": "integer", "optional": True}
                    },
                    "returns": {
                        "rows": {"type": "array", "description": "Query results"},
                        "row_count": {"type": "integer"}
                    }
                },
                "insert": {
                    "description": "Insert records into table",
                    "parameters": {
                        "table": {"type": "string", "description": "Table name"},
                        "records": {"type": "array", "description": "Records to insert"}
                    },
                    "returns": {
                        "inserted_count": {"type": "integer"}
                    }
                },
                "update": {
                    "description": "Update records in table",
                    "parameters": {
                        "table": {"type": "string", "description": "Table name"},
                        "updates": {"type": "object", "description": "Fields to update"},
                        "where": {"type": "object", "description": "Where conditions"}
                    },
                    "returns": {
                        "updated_count": {"type": "integer"}
                    }
                }
            }
        }
    
    async def _execute_query(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query."""
        sql = parameters.get("sql")
        if not sql:
            raise OperationError("sql is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "sql": sql
        }
        
        # Add optional parameters
        if "parameters" in parameters:
            mcp_params["parameters"] = parameters["parameters"]
        
        if "limit" in parameters:
            mcp_params["limit"] = parameters["limit"]
        
        if "timeout" in parameters:
            mcp_params["timeout"] = parameters["timeout"]
        
        response = await self.send_request("database.query", mcp_params)
        
        return {
            "rows": response.result.get("rows", []),
            "row_count": response.result.get("row_count", 0),
            "columns": response.result.get("columns", []),
            "execution_time_ms": response.result.get("execution_time_ms", 0)
        }
    
    async def _insert_records(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Insert records into table."""
        table = parameters.get("table")
        records = parameters.get("records")
        
        if not table:
            raise OperationError("table is required")
        
        if not records:
            raise OperationError("records is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table,
            "records": records
        }
        
        # Add optional parameters
        if "on_conflict" in parameters:
            mcp_params["on_conflict"] = parameters["on_conflict"]
        
        if "return_ids" in parameters:
            mcp_params["return_ids"] = parameters["return_ids"]
        
        response = await self.send_request("database.insert", mcp_params)
        
        return {
            "inserted_count": response.result.get("inserted_count", 0),
            "inserted_ids": response.result.get("inserted_ids", []),
            "failed_records": response.result.get("failed_records", [])
        }
    
    async def _update_records(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update records in table."""
        table = parameters.get("table")
        updates = parameters.get("updates")
        where = parameters.get("where")
        
        if not table:
            raise OperationError("table is required")
        
        if not updates:
            raise OperationError("updates is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table,
            "updates": updates
        }
        
        if where:
            mcp_params["where"] = where
        
        response = await self.send_request("database.update", mcp_params)
        
        return {
            "updated_count": response.result.get("updated_count", 0),
            "matched_count": response.result.get("matched_count", 0)
        }
    
    async def _delete_records(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete records from table."""
        table = parameters.get("table")
        where = parameters.get("where")
        
        if not table:
            raise OperationError("table is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table
        }
        
        if where:
            mcp_params["where"] = where
        else:
            # Safety check - require explicit confirmation for delete all
            if not parameters.get("confirm_delete_all", False):
                raise OperationError("where clause required or confirm_delete_all must be True")
        
        response = await self.send_request("database.delete", mcp_params)
        
        return {
            "deleted_count": response.result.get("deleted_count", 0)
        }
    
    async def _create_table(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new table."""
        table = parameters.get("table")
        schema = parameters.get("schema")
        
        if not table:
            raise OperationError("table is required")
        
        if not schema:
            raise OperationError("schema is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table,
            "schema": schema
        }
        
        # Add optional parameters
        if "if_not_exists" in parameters:
            mcp_params["if_not_exists"] = parameters["if_not_exists"]
        
        if "indexes" in parameters:
            mcp_params["indexes"] = parameters["indexes"]
        
        response = await self.send_request("database.create_table", mcp_params)
        
        return {
            "table": table,
            "created": response.result.get("created", False)
        }
    
    async def _drop_table(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Drop a table."""
        table = parameters.get("table")
        
        if not table:
            raise OperationError("table is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table,
            "if_exists": parameters.get("if_exists", True)
        }
        
        response = await self.send_request("database.drop_table", mcp_params)
        
        return {
            "table": table,
            "dropped": response.result.get("dropped", False)
        }
    
    async def _list_tables(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List all tables in database."""
        mcp_params = {
            "database_name": self.module.config.database_name
        }
        
        if "pattern" in parameters:
            mcp_params["pattern"] = parameters["pattern"]
        
        response = await self.send_request("database.list_tables", mcp_params)
        
        return {
            "tables": response.result.get("tables", []),
            "total_count": response.result.get("total_count", 0)
        }
    
    async def _describe_table(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get table schema information."""
        table = parameters.get("table")
        
        if not table:
            raise OperationError("table is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table
        }
        
        response = await self.send_request("database.describe_table", mcp_params)
        
        return response.result
    
    async def _execute_transaction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple operations in a transaction."""
        operations = parameters.get("operations")
        
        if not operations:
            raise OperationError("operations is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "operations": operations,
            "isolation_level": parameters.get("isolation_level", "READ_COMMITTED")
        }
        
        response = await self.send_request("database.transaction", mcp_params)
        
        return {
            "success": response.result.get("success", False),
            "results": response.result.get("results", []),
            "failed_operation": response.result.get("failed_operation")
        }
    
    async def _bulk_insert(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Bulk insert records efficiently."""
        table = parameters.get("table")
        records = parameters.get("records")
        
        if not table:
            raise OperationError("table is required")
        
        if not records:
            raise OperationError("records is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table,
            "records": records,
            "batch_size": parameters.get("batch_size", 1000),
            "on_conflict": parameters.get("on_conflict", "IGNORE")
        }
        
        response = await self.send_request("database.bulk_insert", mcp_params)
        
        return {
            "inserted_count": response.result.get("inserted_count", 0),
            "failed_count": response.result.get("failed_count", 0),
            "batches_processed": response.result.get("batches_processed", 0)
        }
    
    async def _export_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Export data from table or query."""
        format_type = parameters.get("format", "json")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "format": format_type
        }
        
        if "table" in parameters:
            mcp_params["table"] = parameters["table"]
        elif "sql" in parameters:
            mcp_params["sql"] = parameters["sql"]
        else:
            raise OperationError("Either table or sql must be specified")
        
        if "limit" in parameters:
            mcp_params["limit"] = parameters["limit"]
        
        if "output_path" in parameters:
            mcp_params["output_path"] = parameters["output_path"]
        
        response = await self.send_request("database.export", mcp_params)
        
        return response.result
    
    async def _import_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Import data into table."""
        table = parameters.get("table")
        
        if not table:
            raise OperationError("table is required")
        
        mcp_params = {
            "database_name": self.module.config.database_name,
            "table": table
        }
        
        if "file_path" in parameters:
            mcp_params["file_path"] = parameters["file_path"]
        elif "data" in parameters:
            mcp_params["data"] = parameters["data"]
        else:
            raise OperationError("Either file_path or data must be specified")
        
        # Add optional parameters
        if "format" in parameters:
            mcp_params["format"] = parameters["format"]
        
        if "delimiter" in parameters:
            mcp_params["delimiter"] = parameters["delimiter"]
        
        if "skip_headers" in parameters:
            mcp_params["skip_headers"] = parameters["skip_headers"]
        
        response = await self.send_request("database.import", mcp_params)
        
        return {
            "imported_count": response.result.get("imported_count", 0),
            "failed_count": response.result.get("failed_count", 0),
            "errors": response.result.get("errors", [])
        }
    
    # High-level convenience methods
    
    async def select(self, table: str, columns: List[str] = None, where: Dict[str, Any] = None, 
                    limit: int = None, order_by: str = None) -> List[Dict[str, Any]]:
        """High-level SELECT operation."""
        sql_parts = ["SELECT"]
        
        if columns:
            sql_parts.append(", ".join(columns))
        else:
            sql_parts.append("*")
        
        sql_parts.extend(["FROM", table])
        
        params = []
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            sql_parts.extend(["WHERE", " AND ".join(conditions)])
        
        if order_by:
            sql_parts.extend(["ORDER BY", order_by])
        
        if limit:
            sql_parts.extend(["LIMIT", str(limit)])
        
        sql = " ".join(sql_parts)
        
        result = await self._execute_query({
            "sql": sql,
            "parameters": params if params else None
        })
        
        return result["rows"]
    
    async def insert(self, table: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """High-level INSERT operation for single record."""
        return await self._insert_records({
            "table": table,
            "records": [record]
        })
    
    async def update(self, table: str, updates: Dict[str, Any], 
                    where: Dict[str, Any] = None) -> Dict[str, Any]:
        """High-level UPDATE operation."""
        return await self._update_records({
            "table": table,
            "updates": updates,
            "where": where
        })
    
    async def delete(self, table: str, where: Dict[str, Any] = None, 
                    confirm_delete_all: bool = False) -> Dict[str, Any]:
        """High-level DELETE operation."""
        return await self._delete_records({
            "table": table,
            "where": where,
            "confirm_delete_all": confirm_delete_all
        })