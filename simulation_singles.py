from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.CndbTools import cndbTools
import matplotlib.pyplot as plt
import matplotlib as mpl
import mdtraj as md
import numpy as np
import os  # Para lidar com caminhos
import winsound
import time
cndbTools = cndbTools()

contagem_inicial = time.time()  # Marca o início

def criando_cromatina(sequencia):
    # Criando o arquivo e escrevendo o conteúdo
    with open(sequencia, "w") as file:
        file.write("""1 NA\n2 NA\n3 B1\n4 B1\n5 B1\n6 B1\n7 B1\n8 B1\n9 B1\n10 B1
    11 B1\n12 B1\n13 B1\n14 B1\n15 B1\n16 B1\n17 B1\n18 B1\n19 A1\n20 A1
    21 A1\n22 A1\n23 B1\n24 B1\n25 B1\n26 B1\n27 B1\n28 B1\n29 B1\n30 B1
    31 B1\n32 B1\n33 B1\n34 B1\n35 B1\n36 B1\n37 B2\n38 B2\n39 B2\n40 B2
    41 B2\n42 B2\n43 B2\n44 B2\n45 B2\n46 B2\n47 B2\n48 B2\n49 B2\n50 B2
    51 B2\n52 B2\n53 B2\n54 B2\n55 B2\n56 B2\n57 B2\n58 B2\n59 B2\n60 B2
    61 B1\n62 B1\n63 A2\n64 A2\n65 B1\n66 B1\n67 B2\n68 B2\n69 B2\n70 B2
    71 B1\n72 B1\n73 B2\n74 B2\n75 B1\n76 B1\n77 A2\n78 A2\n79 A2\n80 A2
    81 A2\n82 A2\n83 B1\n84 B1\n85 B2\n86 B2\n87 B2\n88 B2\n89 B2\n90 B2
    91 B2\n92 B2\n93 B2\n94 B2\n95 B2\n96 B2\n97 B2\n98 B2\n99 B2\n100 B2
    101 B2\n102 B2\n103 B2\n104 B2\n105 B2\n106 B2\n107 B2\n108 B2\n109 B1\n110 B1
    111 B1\n112 B1\n113 A2\n114 A2\n115 A2\n116 A2\n117 A1\n118 A1\n119 A1\n120 A1
    121 A2\n122 A2\n123 A2\n124 A2\n125 A2\n126 A2\n127 A2\n128 A2\n129 A2\n130 A2
    131 B1\n132 B1\n133 B1\n134 B1\n135 B1\n136 B1\n137 B2\n138 B2\n139 B1\n140 B1
    141 B2\n142 B2\n143 B2\n144 B2\n145 A2\n146 A2\n147 A2\n148 A2\n149 A2\n150 A2
    151 A2\n152 A2\n153 B2\n154 B2\n155 B2\n156 B2\n157 A2\n158 A2\n159 B2\n160 B2
    161 B2\n162 B2\n163 B2\n164 B2\n165 B2\n166 B2\n167 B2\n168 B2\n169 B2\n170 B2
    171 B2\n172 B2\n173 B2\n174 B2\n175 B2\n176 B2\n177 B2\n178 B2\n179 B2\n180 B2
    181 B2\n182 B2\n183 B2\n184 B2\n185 B2\n186 B2\n187 B2\n188 B2\n189 B2\n190 B2
    191 B2\n192 B2\n193 B2\n194 B2\n195 B2\n196 B2\n197 B2\n198 B2\n199 B2\n200 B2
    201 B2\n202 B2\n203 B2\n204 B2\n205 B2\n206 B2\n207 B2\n208 B2\n209 B2\n210 B2
    211 B2\n212 B2\n213 B2\n214 B2\n215 B2\n216 B2\n217 B2\n218 B2\n219 B2\n220 B2
    221 B2\n222 B2\n223 B2\n224 B2\n225 A2\n226 A2\n227 A2\n228 A2\n229 A2\n230 A2""")

    print("Arquivo 'chr10_beads.txt' criado com sucesso!")

def inicializar_simulacao(nome_cromatina, nome_sequencia, pasta_output):
    # Inicializa a simulação para o cromossomo
    sim_chr = MiChroM(name=nome_cromatina, temperature=1.0, timeStep=0.01)
    sim_chr.setup(platform="cuda")
    sim_chr.saveFolder(pasta_output)

    # Cria a estrutura inicial
    chrom_structure = sim_chr.createRandomWalk(ChromSeq=nome_sequencia)
    sim_chr.loadStructure(chrom_structure, center=True)

    # Adiciona forças e potenciais
    sim_chr.addFENEBonds(kFb=30.0)
    sim_chr.addAngles(kA=2.0)
    sim_chr.addRepulsiveSoftCore(eCut=4.0)
    sim_chr.addFlatBottomHarmonic(nRad=9)
    sim_chr.addTypetoType(mu=3.22, rc=1.78)
    sim_chr.addIdealChromosome(mu=3.22, rc=1.78, dinit=3, dend=500)
    return sim_chr

def rodar_simulacao_colapso(sim_chr, block):
    print("Colapsando o cromossomo...")
    sim_chr.createSimulation()
    sim_chr.run(block)
    return sim_chr

def rodar_simulacao_final(sim_chr, block):
    sim_chr.removeFlatBottomHarmonic()
    sim_chr.createReporters(statistics=True, traj=True, trajFormat="cndb")
    sim_chr.run(block)
    return sim_chr

def analise_hi_c_densidade(folder, arquivo_trajetoria, indice):
    """Gera e salva a matriz Hi-C como .png e .dense na pasta especificada."""

    # Nome do arquivo de densidade radial dentro da pasta
    file_path_txt = os.path.join(folder, "densidade_radial_" + str(indice) + ".txt")


    # Criar a pasta se não existir
    os.makedirs(folder, exist_ok=True)

    # Nome dos arquivos dentro da pasta
    file_image = os.path.join(folder, 'hi_c_matrix_' + str(indice) + '.png')
    file_dense = os.path.join(folder, 'hi_c_matrix_' + str(indice) + '.dense')

    # Etapa de análise de resultados
    chr10_traj = cndbTools.load(arquivo_trajetoria)
    first_frame = min([int(key) for key in chr10_traj.cndb.keys() if key != 'types'])
    last_frame = max([int(key) for key in chr10_traj.cndb.keys() if key != 'types'])
    chr10_xyz = cndbTools.xyz(frames=[first_frame, last_frame, 1], XYZ=[0, 1, 2])

    print(f"First frame: {first_frame}.\nLast frame: {last_frame}.")
    print("Generating the contact probability matrix...")

    # Gerando Hi-C
    chr10_sim_HiC = cndbTools.traj2HiC(chr10_xyz)

    # Salvar a matriz Hi-C em formato .png
    plt.matshow(chr10_sim_HiC, norm=mpl.colors.LogNorm(vmin=0.01, vmax=chr10_sim_HiC.max()), cmap="Reds")
    plt.colorbar()
    plt.savefig(file_image, dpi=400, bbox_inches="tight")
    plt.close()  # Fecha a figura para liberar memória
    print(f"Imagem Hi-C salva em {file_image}")

    # Salvar a matriz Hi-C no formato .dense
    np.savetxt(file_dense, chr10_sim_HiC, fmt="%.6f", delimiter=" ")
    print(f"Arquivo Hi-C no formato .dense salvo em {file_dense}")
    
    chr10_A1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'A1'], XYZ=[0,1,2])
    chr10_A2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'A2'], XYZ=[0,1,2])
    chr10_B1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'B1'], XYZ=[0,1,2])
    chr10_B2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'B2'], XYZ=[0,1,2])
    chr10_NA = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'NA'], XYZ=[0,1,2])

    # Função para filtrar índices inválidos
    def filtra_indices(beadSelection, total_beads):
        return [i for i in beadSelection if i < total_beads]

    # Parâmetros iniciais
    radius = 15.0
    bins = 200

    # Obtenção dos dados para A1 e B1
    # Substitua `chr10_traj.dictChromSeq[b'A1']` e `chr10_traj.dictChromSeq[b'B1']` pelos dados corretos
    beadSelection_A1 = chr10_traj.dictChromSeq[b'A1']
    beadSelection_A2 = chr10_traj.dictChromSeq[b'A2']
    beadSelection_B1 = chr10_traj.dictChromSeq[b'B1']
    beadSelection_B2 = chr10_traj.dictChromSeq[b'B2']
    beadSelection_NA = chr10_traj.dictChromSeq[b'NA']

    #"b'A2', b'NA', b'B3', b'B1', b'A1', b'B2'"
    # Obtendo os arrays tridimensionais
    chr10_A1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_A1, XYZ=[0, 1, 2])
    chr10_A2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_A2, XYZ=[0, 1, 2])
    chr10_B1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_B1, XYZ=[0, 1, 2])
    chr10_B2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_B2, XYZ=[0, 1, 2])
    chr10_NA = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_NA, XYZ=[0, 1, 2])

    # Filtra os índices inválidos
    beadSelection_A1_validos = filtra_indices(beadSelection_A1, chr10_A1.shape[1])
    beadSelection_A2_validos = filtra_indices(beadSelection_A2, chr10_A1.shape[1])
    beadSelection_B1_validos = filtra_indices(beadSelection_B1, chr10_B1.shape[1])
    beadSelection_B2_validos = filtra_indices(beadSelection_B2, chr10_B1.shape[1])
    beadSelection_NA_validos = filtra_indices(beadSelection_NA, chr10_NA.shape[1])

    # Calculando o RDP para A1
    r_A1, RDP_chr10_A1 = cndbTools.compute_RDP(
        chr10_A1, beadSelection=beadSelection_A1_validos, radius=radius, bins=bins
    )

    # Calculando o RDP para A2
    r_A2, RDP_chr10_A2 = cndbTools.compute_RDP(
        chr10_A2, beadSelection=beadSelection_A2_validos, radius=radius, bins=bins
    )

    # Calculando o RDP para B1
    r_B1, RDP_chr10_B1 = cndbTools.compute_RDP(
        chr10_B1, beadSelection=beadSelection_B1_validos, radius=radius, bins=bins
    )

    # Calculando o RDP para B2
    r_B2, RDP_chr10_B2 = cndbTools.compute_RDP(
        chr10_B2, beadSelection=beadSelection_B2_validos, radius=radius, bins=bins
    )
    # Calculando o RDP para NA
    r_NA, RDP_chr10_NA = cndbTools.compute_RDP(
    chr10_NA, beadSelection=beadSelection_NA_validos, radius=radius, bins=bins
    )


    plt.figure(figsize=(12, 5))
    plt.plot(r_A1, RDP_chr10_A1, color='red', label='A1')
    plt.plot(r_A2, RDP_chr10_A2, color='purple', label='A2')
    plt.plot(r_B1, RDP_chr10_B1, color='blue', label='B1')
    plt.plot(r_B2, RDP_chr10_B2, color='green', label='B2')
    plt.plot(r_NA, RDP_chr10_NA, color='yellow', label='NA')
    plt.xlabel(r'r ($\sigma$)', fontsize=20,fontweight='normal', color='k')
    plt.ylabel(r'$\rho(r)/N_{type}$', fontsize=20,fontweight='normal', color='k')
    plt.legend()
    plt.gca().set_xlim(xmin=0,xmax=9)

    # Nome do arquivo dentro da pasta
    file_path_densidade = os.path.join(folder, "densidade_radial_" + str(indice) + '.png')

    # Salvar dentro da pasta
    plt.savefig(file_path_densidade, dpi=400)
    plt.close()  # Fecha a figura corretamente
    print(f"Imagem Hi-C salva em {file_path_densidade}")

    # Salvar os dados de densidade radial no formato .txt
    np.savetxt(file_path_txt, np.column_stack((r_A1, RDP_chr10_A1, r_A2, RDP_chr10_A2, r_B1, RDP_chr10_B1, r_B2, RDP_chr10_B2, r_NA, RDP_chr10_NA)), fmt="%.6f", delimiter=" ", header="r_A1 RDP_A1 r_A2 RDP_A2 r_B1 RDP_B1 r_B2 RDP_B2 r_NA RDP_NA")

    print(f"Arquivo de densidade radial salvo em {file_path_txt}")

def analise_statistics(folder_input, indice, folder='resultado'):
    """Gera e salva gráficos para as variáveis RG, Temperature e Etotal na pasta especificada."""

    # Criar a pasta se não existir
    os.makedirs(folder, exist_ok=True)

    # Carregar dados do arquivo ignorando comentários (#)
    data = np.genfromtxt(folder_input, delimiter=',', comments='#', names=True)

    colunas = ["RG", "Temperature", "Etotal"]  # Lista das variáveis a serem analisadas
    
    for coluna in colunas:
        if coluna in data.dtype.names:  # Verifica se a coluna existe no arquivo
            steps = data["Step"]
            valores = data[coluna]

            # Criar o gráfico
            plt.figure(figsize=(12, 5))
            plt.plot(steps, valores, linestyle='-', color='blue')
            plt.xlabel("Step", fontsize=12)
            plt.ylabel(coluna, fontsize=12)
            plt.title(coluna, fontsize=14)
            plt.grid(True)
            plt.tight_layout()

            # Caminho para salvar o gráfico
            file_output = os.path.join(folder, f"{coluna}_" + str(indice) + '.png')

            # Salvar gráfico na pasta especificada
            plt.savefig(file_output, dpi=400, bbox_inches="tight")
            print(f"Gráfico de {coluna} salvo em {file_output}")

            plt.close()  # Fechar o gráfico para evitar consumo excessivo de memória

def salvar_trajetoria(pasta_output, nome_cromatina, arquivo_trajetoria):
    # Etapa de análise de resultados
    chr10_traj = cndbTools.load(arquivo_trajetoria)
    first_frame = min([int(key) for key in chr10_traj.cndb.keys() if key != 'types'])
    last_frame = max([int(key) for key in chr10_traj.cndb.keys() if key != 'types'])
    chr10_xyz = cndbTools.xyz(frames=[first_frame, last_frame + 1, 1], XYZ=[0, 1, 2])

    # Definir o caminho de saída para os arquivos
    path_to_cndb = pasta_output + '/'
    file_name = nome_cromatina + '_0'

    # Mapeamento dos tipos de resíduos para o formato PDB
    Type_conversion = {'A1': 'ASP', 'A2': 'GLU', 'B1': 'LYS', 'B2': 'ARG', 'B3': 'HIS', 'B4': 'HIS', 'NA': 'GLY'}

    # Selecionar o primeiro frame para o PDB
    frame = 0  # Ajuste conforme necessário

    # Gerar o arquivo PDB
    pdb_path = f"{path_to_cndb}{file_name}.pdb"
    with open(pdb_path, "w") as f:
        for i in range(len(chr10_xyz[frame])):
            j = ['' for _ in range(9)]
            j[0] = 'ATOM'.ljust(6)
            j[1] = str(i + 1).rjust(5)
            j[2] = 'CA'.center(4)

            # Corrigir conversão de bytes para string antes de acessar Type_conversion
            residue_name = chr10_traj.ChromSeq[i].decode("utf-8") if isinstance(chr10_traj.ChromSeq[i], bytes) else chr10_traj.ChromSeq[i]
            j[3] = Type_conversion.get(residue_name, 'GLY').ljust(3)  # Usa 'GLY' como padrão caso não esteja no dicionário

            j[4] = 'A'.rjust(1)
            j[5] = str(i + 1).rjust(4)
            j[6] = str('%8.3f' % float(chr10_xyz[frame][i][0])).rjust(8)
            j[7] = str('%8.3f' % float(chr10_xyz[frame][i][1])).rjust(8)
            j[8] = str('%8.3f' % float(chr10_xyz[frame][i][2])).rjust(8)
            f.write("{}{} {} {} {}{}    {}{}{}\n".format(*j))
        f.write("END\n")

    # Carregar e salvar a trajetória em XTC
    traj = md.load(pdb_path)
    traj.xyz = chr10_xyz
    traj.time = np.linspace(0, len(chr10_xyz), len(chr10_xyz))
    traj.save_xtc(f"{path_to_cndb}{file_name}.xtc")

    print("Arquivos PDB e XTC gerados com sucesso!")   

def calcular_media_dense(folder="resultado", output_file="hi_c_matrix_combinado.dense"):
    """Calcula a média das matrizes Hi-C (.dense) e salva um novo arquivo combinado."""

    # Listar todos os arquivos .dense na pasta
    files = sorted([f for f in os.listdir(folder) if f.startswith("hi_c_matrix_") and f.endswith(".dense")])
    
    if not files:
        print("Nenhum arquivo .dense encontrado na pasta.")
        return

    print(f"Encontrados {len(files)} arquivos para combinar.")

    # Carregar todas as matrizes
    matrices = [np.loadtxt(os.path.join(folder, f)) for f in files]

    # Verificar se todas as matrizes têm o mesmo tamanho
    shapes = [mat.shape for mat in matrices]
    if len(set(shapes)) > 1:
        print("Erro: As matrizes Hi-C têm tamanhos diferentes e não podem ser combinadas.")
        return

    # Calcular a média das matrizes
    hi_c_combined = np.mean(matrices, axis=0)

    # Salvar o resultado
    output_path = os.path.join(folder, output_file)
    np.savetxt(output_path, hi_c_combined, fmt="%.6f", delimiter=" ")
    print(f"Arquivo combinado salvo em {output_path}")

def analise_hi_c(folder='resultado', arquivo_dense='resultado/hi_c_matrix_combinado.dense'):
    """Gera e salva a matriz Hi-C a partir de um arquivo .dense, além de um gráfico de distribuição radial."""

    # Criar a pasta se não existir
    os.makedirs(folder, exist_ok=True)

    # Nome dos arquivos dentro da pasta
    file_image = os.path.join(folder, f"hi_c_matrix_combinacao.png")

    # Carregar matriz Hi-C do arquivo .dense
    hic_matrix = np.loadtxt(arquivo_dense)

    # Verificar dimensões
    print(f"Matriz Hi-C carregada com formato: {hic_matrix.shape}")

    # Gerar e salvar o mapa Hi-C
    plt.matshow(hic_matrix, norm=mpl.colors.LogNorm(vmin=0.005, vmax=hic_matrix.max()), cmap="Reds")
    plt.colorbar()
    plt.savefig(file_image, dpi=400, bbox_inches="tight")
    print(f"Imagem Hi-C salva em {file_image}")

def calcular_media_densidade_radial(folder='resultado'):
    """Calcula a média da densidade radial corretamente e copia as colunas de raio sem alteração."""

    # Criar a pasta se não existir
    os.makedirs(folder, exist_ok=True)

    # Buscar automaticamente os arquivos que seguem o padrão 'densidade_radial_*.txt'
    arquivos = sorted([f for f in os.listdir(folder) if f.startswith('densidade_radial_') and f.endswith('.txt')])

    if not arquivos:
        print("Nenhum arquivo de densidade radial encontrado. Verifique a pasta.")
        return

    # Lista para armazenar os dados das réplicas
    dados_matriz = []

    for arquivo in arquivos:
        caminho = os.path.join(folder, arquivo)
        try:
            dados_arquivo = np.loadtxt(caminho)
            dados_matriz.append(dados_arquivo)
        except Exception as e:
            print(f"Erro ao carregar {arquivo}: {e}")

    if not dados_matriz:
        print("Erro ao carregar os dados de densidade radial.")
        return

    # Converter para matriz NumPy
    dados_matriz = np.array(dados_matriz)

    # Pegando os valores de raio diretamente (copiando sem alteração)
    r_A1 = dados_matriz[0][:, 0]
    r_A2 = dados_matriz[0][:, 2]
    r_B1 = dados_matriz[0][:, 4]
    r_B2 = dados_matriz[0][:, 6]
    r_NA = dados_matriz[0][:, 8]

    # Calculando a média das colunas de RDP
    RDP_A1_media = np.mean(dados_matriz[:, :, 1], axis=0)
    RDP_A2_media = np.mean(dados_matriz[:, :, 3], axis=0)
    RDP_B1_media = np.mean(dados_matriz[:, :, 5], axis=0)
    RDP_B2_media = np.mean(dados_matriz[:, :, 7], axis=0)
    RDP_NA_media = np.mean(dados_matriz[:, :, 9], axis=0)

    # Nome dos arquivos de saída
    file_media_txt = os.path.join(folder, "media_densidade_radial.txt")
    file_media_png = os.path.join(folder, "media_densidade_radial.png")

    # Salvar os dados médios em um arquivo .txt (mantendo as colunas de raio intactas)
    np.savetxt(file_media_txt, np.column_stack((r_A1, RDP_A1_media, r_A2, RDP_A2_media, r_B1, RDP_B1_media, r_B2, RDP_B2_media, r_NA, RDP_NA_media)), 
               fmt="%.6f", delimiter=" ", header="r_A1 RDP_A1 r_A2 RDP_A2 r_B1 RDP_B1 r_B2 RDP_B2 r_NA RDP_NA")
    
    print(f"Dados de densidade radial média salvos em {file_media_txt}")

    # Gerar gráfico com o eixo X = Raio e eixo Y = Médias das RDPs
    plt.figure(figsize=(10, 4))
    plt.plot(r_A1, RDP_A1_media, color='red', label='RDP_A1')
    plt.plot(r_A2, RDP_A2_media, color='purple', label='RDP_A2')
    plt.plot(r_B1, RDP_B1_media, color='blue', label='RDP_B1')
    plt.plot(r_B2, RDP_B2_media, color='green', label='RDP_B2')
    plt.plot(r_NA, RDP_NA_media, color='yellow', label='RDP_NA')
    
    plt.xlabel("Raio", fontsize=14)
    plt.ylabel("Densidade Média", fontsize=14)
    plt.title("Distribuição Radial Média", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_media_png, dpi=400)
    
    print(f"Gráfico da densidade radial média salvo em {file_media_png}")


def notificar_som():
    winsound.Beep(970, 900)  # Toca um som de 970 Hz por 500ms

if __name__ == "__main__":
    # Sequência da cromatina
    nome_sequencia = 'chr10_beads.txt'
    pasta_resultado = 'resultado'
    criando_cromatina(nome_sequencia)

    # Quantidade de simulações
    replicas = 3
    print(f'--------------------------------------------------------------------------------------- Simulação de {replicas} réplicas ---------------------------------------------------------------------------------------')
    for indice in range(replicas):
        # Variáveis
        nome_cromatina = 'chr10' + '_' + str(indice)
        pasta_output = 'output_chr10' + '_' + str(indice)
        arquivo_trajetoria = pasta_output + '/' + nome_cromatina + '_0.cndb'
        arquivo_statistics = pasta_output + '/' + 'statistics.txt'
        
        
        # Fase 1: Inicializar simulação (parâmetos da simulação)
        sim_10 = inicializar_simulacao(nome_cromatina,
                                        nome_sequencia,
                                        pasta_output)
        
        # Fase 2: Rodar o colapso
        estrutura_colapsada = rodar_simulacao_colapso(sim_10, 150*10**3)
        print(f'-------------------------------------------------------------------------------- Cromatina {nome_cromatina} colapsada --------------------------------------------------------------------------------')

        # Fase 3: Rodar simulação final
        rodar_simulacao_final(estrutura_colapsada, 300*10**3)
        print(f'-------------------------------------------------------------------------------- Cromatina {nome_cromatina} simulada --------------------------------------------------------------------------------')

        # Analises das simulações
        analise_hi_c_densidade(pasta_resultado, arquivo_trajetoria, indice)                             
        analise_statistics(arquivo_statistics, indice)
        salvar_trajetoria(pasta_output, nome_cromatina, arquivo_trajetoria)

    # Análise de combinação dos dados das simulações
    calcular_media_dense()
    analise_hi_c()
    calcular_media_densidade_radial()

    # Fim das simulações
    notificar_som()

contagem_final = time.time()  # Marca o fim
tempo = contagem_final - contagem_inicial  # Calcula a duração


print(f"Tempo de execução: {tempo:.2f} segundos")