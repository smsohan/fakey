Create a Docker image and run it on Cloud Run using GPU

This is a Dockerized version of [insightface swapper](https://github.com/deepinsight/insightface/tree/master/examples/in_swapper)

Update [cloudbuild.yaml](./cloudbuild.yaml) and then build the container.
```bash
$ gcloud builds submit --config cloudbuild.yaml --region <REGION>
```
