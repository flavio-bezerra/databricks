from typing import Dict
from azure.identity import ClientSecretCredential

def connect_jdbc(tenant_id: str, client_id: str, client_secret: str, scope: str) -> Dict[str, str]:
    """
    Retorna um dicionário de propriedades de conexão JDBC usando autenticação Azure AD.

    Args:
        tenant_id (str): ID do tenant do Azure AD.
        client_id (str): ID do aplicativo registrado no Azure AD.
        client_secret (str): Segredo do aplicativo registrado.
        scope (str): Escopo para o token de acesso.

    Returns:
        Dict[str, str]: Propriedades de conexão JDBC contendo 'accessToken' e 'driver'.
    """
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    token = credential.get_token(scope).token
    return {
        "accessToken": token,  
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    }

def url_jdbc(sql_endpoint: str, database: str) -> str:
    """
    Monta a URL JDBC para conexão com o Azure SQL Database.

    Args:
        sql_endpoint (str): Endpoint do SQL Server.
        database (str): Nome do banco de dados.

    Returns:
        str: URL JDBC formatada.
    """
    return f"jdbc:sqlserver://{sql_endpoint};database={database};encrypt=false;trustServerCertificate=true;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"
