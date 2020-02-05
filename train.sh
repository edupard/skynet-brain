export COMMIT_SHA=`git rev-parse HEAD`
export PROJECT_ID=skynet-1984
export IMAGE_REPO_NAME=nn
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$COMMIT_SHA
export REGION=us-central1
export JOB_NAME=test_7
echo $IMAGE_URI

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --scale-tier BASIC_TPU \
  --tpu-tf-version 1.15 \
  -- \
  --model-idx=1