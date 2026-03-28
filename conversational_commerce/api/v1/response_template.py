class ResponseTemplate:
    """Response template."""    
    @staticmethod
    def _success_response_template(data: dict) -> dict:
        return {
            "data": data,
            "status": True,
            "message": "Successful",
        }
    
    @staticmethod
    def _error_response_template(message: str) -> dict:
        return {
            "data": None,
            "status": False,
            "message": message,
        }