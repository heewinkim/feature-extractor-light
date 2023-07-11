FROM public.ecr.aws/lambda/python:3.8

RUN yum install unzip wget expat curl glibc -y

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY resnet18.pth ${LAMBDA_TASK_ROOT}
COPY app.py ${LAMBDA_TASK_ROOT}

CMD [ "app.lambda_handler" ] 
