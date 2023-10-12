

Install dependencies:
```shell
make install
```

Run bot locally:
```shell
make run
```

Run bot on Beam:
```shell
make run_beam
```

Deploy the bot under a RESTful API using Beam:
```shell
make deploy_beam
```

Make a request to the bot calling the RESTful API:
```shell
export BEAM_DEPLOYMENT_ID=<BEAM_DEPLOYMENT_ID>
export BEAM_AUTH_TOKEN=<BEAM_AUTH_TOKEN>

make call_restful_api DEPLOYMENT_ID=${BEAM_DEPLOYMENT_ID} TOKEN=${BEAM_AUTH_TOKEN} 
```