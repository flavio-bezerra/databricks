import mlflow
import pickle
import pandas as pd
import numpy as np

class DartsWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper melhorado para suportar inferencia com covariaveis globais.
    """
    def load_context(self, context):
        with open(context.artifacts["darts_model"], "rb") as f:
            self.model = pickle.load(f)
        
        self.future_covariates = None
        if "future_covariates" in context.artifacts:
            try:
                with open(context.artifacts["future_covariates"], "rb") as f:
                    self.future_covariates = pickle.load(f)
            except Exception as e:
                print(f"⚠️ [Wrapper] Erro ao carregar covariáveis: {e}")

    def predict(self, context, model_input):
        n = 1
        if isinstance(model_input, pd.DataFrame):
            if 'n' in model_input.columns:
                try:
                    n = int(model_input.iloc[0]['n'])
                except:
                    pass
        
        predict_kwargs = {"n": n}
        if self.model.supports_future_covariates and self.future_covariates is not None:
             predict_kwargs["future_covariates"] = self.future_covariates

        try:
            pred = self.model.predict(**predict_kwargs)
            if isinstance(pred, list):
                 return pd.concat([p.pd_dataframe() for p in pred])
            return pred.pd_dataframe()
        except Exception as e:
            print(f"⚠️ [Wrapper] Falha na predição: {str(e)}. Retornando Dummy.")
            dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
            return pd.DataFrame(np.zeros((n, 1)), index=dates, columns=["dummy_prediction"])
