pipeline {
    agent any

    stages {
        stage('Preparar') {
            steps {
                sh 'mkdir -p outputs'
            }
        }

        stage('Verificar Docker') {
            steps {
                sh 'docker --version'
                sh 'docker ps'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ml_project .'
            }
        }

        stage('Run Container') {
            steps {
                sh '''
                docker run --rm \
                  -v $(pwd)/outputs:/app/outputs \
                  ml_project
                '''
            }
        }

        stage('Guardar resultados') {
            steps {
                archiveArtifacts artifacts: 'outputs/*', fingerprint: true
            }
        }
    }

    post {
        success {
            echo '✅ Todo funcionó correctamente'
        }
        failure {
            echo '❌ Algo falló, revisa los logs'
        }
    }
}
