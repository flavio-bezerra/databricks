"""
Módulo de Conectores (Ingestion).

Este arquivo contém funções utilitárias para estabelecer conexões seguras com bancos de dados,
especificamente focando no Azure SQL Database via JDBC com autenticação Azure Active Directory (AAD).

Funções:
- connect_jdbc: Gera as credenciais e configurações do driver JDBC.
- url_jdbc: Constrói a string de conexão (Connection String) formatada corretamente.
"""

from typing import Dict
from azure.identity import ClientSecretCredential

def connect_jdbc(tenant_id: str, client_id: str, client_secret: str, scope: str) -> Dict[str, str]:
    """
    Cria e retorna as propriedades de conexão JDBC utilizando autenticação via Service Principal (Azure AD).
    
    Esta função é essencial para conectar ao Azure SQL sem usar usuário/senha fixos no código,
    utilizando tokens de acesso seguros gerados dinamicamente via `ClientSecretCredential`.

    Args:
        tenant_id (str): ID do Diretório (Tenant) do Azure AD onde o app está registrado.
        client_id (str): ID da Aplicação (Client ID) do Service Principal.
        client_secret (str): Segredo (Secret) gerado para a aplicação.
        scope (str): Escopo de permissão para o token (geralmente 'https://database.windows.net/.default').

    Returns:
        Dict[str, str]: Dicionário contendo:
            - 'accessToken': O token JWT gerado para autenticação.
            - 'driver': O nome da classe do driver JDBC do SQL Server.
    """
    # Cria o objeto de credencial usando as chaves do Service Principal
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    
    # Solicita um token de acesso para o escopo definido
    token = credential.get_token(scope).token
    
    # Retorna as propriedades configuradas para o Spark usar na leitura JDBC
    return {
        "accessToken": token,  
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    }

def url_jdbc(sql_endpoint: str, database: str) -> str:
    """
    Constrói a URL de conexão JDBC formatada para o Azure SQL Database.

    Esta string define onde o Spark deve conectar e quais parâmetros de segurança usar.

    Parâmetros da URL configurados:
    - encrypt=false: (Ajustável) Define se a criptografia SSL é obrigatória.
    - trustServerCertificate=true: Permite certificados auto-assinados (comum em ambientes de dev/privados).
    - hostNameInCertificate: Validação extra para garantir que estamos falando com o servidor Azure correto.
    - loginTimeout: Tempo máximo de espera para conectar.

    Args:
        sql_endpoint (str): Endereço do servidor (ex: servidor.database.windows.net).
        database (str): Nome do banco de dados específico.

    Returns:
        str: A string de conexão JDBC completa.
    """
    return f"jdbc:sqlserver://{sql_endpoint};database={database};encrypt=false;trustServerCertificate=true;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"
