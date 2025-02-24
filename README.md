# Songs Recommendation

Este projeto gera recomendações de músicas que o usuário possivelmente irá gostar baseado nas músicas passadas como parâmetros. Uma API é responsável por responder as requisições, e um modelo de Machine Learning foi treinado, baseado num dataset do sportify de 2023, contendo diversas informações das tracks mais tocadas, assim como playlists mais salvas neste ano.

Tudo foi conteinerizado, e configurado para o deploy ocorrer e ser orquestrado pelo Kubernetes. Além disso, há uma integração com Github Actions para detectar mudanças no código e realizar o deployment dos serviços atualizados no Kubernetes.

Trabalho desenvolvido com o intuito de se praticar todo o conteúdo aprendido na disciplina de Computação em Nuvem da UFMG, em especial conceitos de CI/CD, conteinerização, orquestração de containers e integração com ferramentas como GIthub Actions e ArgoCD.

## Módulo de Machine Learning

Um sistema de recomendação de músicas foi desenvolvido baseado em regras de associação. Foi aplicado um algoritmo de Frequent Itemset Mining (FIM), para encontrar padrões nas playlists mais populares, e verificar músicas que se "relacionam" entre si.

O módulo de recomendação foi desenvolvido utilizando o framework __FP-Growth__. 

Os detalhes de implementação podem ser vistos no diretório /recommendation-service, recommendation.py.

Está devidamente conteinerizado, com um arquivo de requirementes e a Dockerfile para construir a imagem. Existe um persistent-volume configurado no kubernetes do servidor, no qual o arquivo de recomendações será salvo, e poderá ser atualizado de acordo com atualizações nos arquivos de recomendações (dataset de músicas e playlists). Além disso, o PV permite que o módulo da API Rest possa acessar o arquivo também.

## API Rest

Há uma API Rest desenvolvida em Pyhton, utilizando o framework Flask. A API consome o arquivo de recomendações de músicas gerado pelo módulo de treinamento, armazenado no Persitent Volume; e responde à requisições recebidas.

Um exemplo de request seria:

```
curl -X POST http://10.43.62.138:52043/api/recommend 
-H "Content-Type: application/json" 
-d '{"songs":["Yesterday", "Bohemian Rhapsody"]}'
```

recebendo-se a seguinte resposta:
```
{
  "model_date": "2025-01-07 22:08:31",
  "songs": [
    "F**kin' Problems",
    "Body Like A Back Road",
    "House Party",
    "One Dance",
    "It Wasn't Me",
    "No Hands (feat. Roscoe Dash and Wale) - Explicit Album Version",
    "Closer",
    "Ignition - Remix",
    "Alright",
    "Too Good"
  ],
  "version": "1.2"
}
```

## Script de testes

Existe também um módulo que irá executar um script python que gera requisições aleatórias para a API, para validar seu funcionamento.

Consome-se um dataset de listagem de músicas aleatórias e gera requisições aleatórias para a API, e as salva no arquivo responses.txt.

## Deployment e Orquestração

Tudo foi devidamente conteinerizado, e configurado para ser orquestrado pelo Kubernetes do servidor. 

Os arquivos .yaml na raiz do projeto descrevem o serviço, o claim do PV existente, e o deployment de cada um dos módulos. É possível aplicá-los no Kubernetes por meio dos comandos:
```
kubectl apply -f deployment-api.yaml
kubectl apply -f deployment-training-model.yaml
kubect apply -f pvc.yaml
kubectl apply -f service.yaml
```

Dessa forma, o Kubernetes irá realizar o deployment dos serviços e estarão disponíveis para serem acessados.

## CI/CD

Foi implementado por meio do github actions, um job que verifica por alterações no código da API, ou então no programa que treina o modelo, e automaticamente pega a versão atual que está rodando que está armazenado num arquivo .env, a incrementa, builda uma nova imagem Docker com essa nova tag, e também automaticamente ajusta no arquivo deployment para apontar para a nova versão.

Por meio disso, existe um fluxo contínuo de CD, uma vez que o commit no código da aplicação, leva a uma nova docker image, e ao deploy dessa nova imagem.
