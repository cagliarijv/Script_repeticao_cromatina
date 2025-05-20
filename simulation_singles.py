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
        file.write("""1 A1\n2 A2\n3 B1\n4 B2\n5 A1\n6 A2\n7 B1\n8 B2\n9 A1\n10 A2
    11 B1\n12 B2\n13 A1\n14 A2\n15 B1\n16 B2\n17 A1\n18 A2\n19 B1\n20 B2
    21 A1\n22 A2\n23 B1\n24 B2\n25 A1\n26 A2\n27 B1\n28 B2\n29 A1\n30 A2
    31 B1\n32 B2\n33 A1\n34 A2\n35 B1\n36 B2\n37 A1\n38 A2\n39 B1\n40 B2
    41 A1\n42 A2\n43 B1\n44 B2\n45 A1\n46 A2\n47 B1\n48 B2\n49 A1\n50 A2
    51 B1\n52 B2\n53 A1\n54 A2\n55 B1\n56 B2\n57 A1\n58 A2\n59 B1\n60 B2
    61 A1\n62 A2\n63 B1\n64 B2\n65 A1\n66 A2\n67 B1\n68 B2\n69 A1\n70 A2
    71 B1\n72 B2\n73 A1\n74 A2\n75 B1\n76 B2\n77 A1\n78 A2\n79 B1\n80 B2
    81 A1\n82 A2\n83 B1\n84 B2\n85 A1\n86 A2\n87 B1\n88 B2\n89 A1\n90 A2
    91 B1\n92 B2\n93 A1\n94 A2\n95 B1\n96 B2\n97 A1\n98 A2\n99 B1\n100 B2
    101 A1\n102 A2\n103 B1\n104 B2\n105 A1\n106 A2\n107 B1\n108 B2\n109 A1\n110 A2
    111 B1\n112 B2\n113 A1\n114 A2\n115 B1\n116 B2\n117 A1\n118 A2\n119 B1\n120 B2
    121 A1\n122 A2\n123 B1\n124 B2\n125 A1\n126 A2\n127 B1\n128 B2\n129 A1\n130 A2
    131 B1\n132 B2\n133 A1\n134 A2\n135 B1\n136 B2\n137 A1\n138 A2\n139 B1\n140 B2
    141 A1\n142 A2\n143 B1\n144 B2\n145 A1\n146 A2\n147 B1\n148 B2\n149 A1\n150 A2
    151 B1\n152 B2\n153 A1\n154 A2\n155 B1\n156 B2\n157 A1\n158 A2\n159 B1\n160 B2
    161 A1\n162 A2\n163 B1\n164 B2\n165 A1\n166 A2\n167 B1\n168 B2\n169 A1\n170 A2
    171 B1\n172 B2\n173 A1\n174 A2\n175 B1\n176 B2\n177 A1\n178 A2\n179 B1\n180 B2
    181 A1\n182 A2\n183 B1\n184 B2\n185 A1\n186 A2\n187 B1\n188 B2\n189 A1\n190 A2
    191 B1\n192 B2\n193 A1\n194 A2\n195 B1\n196 B2\n197 A1\n198 A2\n199 B1\n200 B2
    201 A1\n202 A2\n203 B1\n204 B2\n205 A1\n206 A2\n207 B1\n208 B2\n209 A1\n210 A2
    211 B1\n212 B2\n213 A1\n214 A2\n215 B1\n216 B2\n217 A1\n218 A2\n219 B1\n220 B2
    221 A1\n222 A2\n223 B1\n224 B2\n225 A1\n226 A2\n227 B1\n228 B2\n229 A1\n230 A2""")

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
    sim_chr.createReporters(statistics=True, traj=True, trajFormat="pdb")
    sim_chr.run(block)
    return sim_chr

def analise_hi_c_densidade(folder, arquivo_trajetoria, indice):
    """Gera e salva a matriz Hi-C como .png e .dense na pasta especificada."""

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
    plt.savefig(file_image, dpi=300, bbox_inches="tight")
    plt.close()  # Fecha a figura para liberar memória
    print(f"Imagem Hi-C salva em {file_image}")

    # Salvar a matriz Hi-C no formato .dense
    np.savetxt(file_dense, chr10_sim_HiC, fmt="%.6f", delimiter=" ")
    print(f"Arquivo Hi-C no formato .dense salvo em {file_dense}")
    
    chr10_A1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'A1'], XYZ=[0,1,2])
    chr10_A2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'A2'], XYZ=[0,1,2])
    chr10_B1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'B1'], XYZ=[0,1,2])
    chr10_B2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=chr10_traj.dictChromSeq[b'B2'], XYZ=[0,1,2])

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

    #"b'A2', b'NA', b'B3', b'B1', b'A1', b'B2'"
    # Obtendo os arrays tridimensionais
    chr10_A1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_A1, XYZ=[0, 1, 2])
    chr10_A2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_A2, XYZ=[0, 1, 2])
    chr10_B1 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_B1, XYZ=[0, 1, 2])
    chr10_B2 = cndbTools.xyz(frames=list(range(first_frame, last_frame + 1)), beadSelection=beadSelection_B2, XYZ=[0, 1, 2])

    # Filtra os índices inválidos
    beadSelection_A1_validos = filtra_indices(beadSelection_A1, chr10_A1.shape[1])
    beadSelection_A2_validos = filtra_indices(beadSelection_A2, chr10_A1.shape[1])
    beadSelection_B1_validos = filtra_indices(beadSelection_B1, chr10_B1.shape[1])
    beadSelection_B2_validos = filtra_indices(beadSelection_B2, chr10_B1.shape[1])

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

    plt.figure(figsize=(12, 5))
    plt.plot(r_A1, RDP_chr10_A1, color='red', label='A1')
    plt.plot(r_A2, RDP_chr10_A2, color='purple', label='A2')
    plt.plot(r_B1, RDP_chr10_B1, color='blue', label='B1')
    plt.plot(r_B2, RDP_chr10_B2, color='green', label='B2')
    plt.xlabel(r'r ($\sigma$)', fontsize=20,fontweight='normal', color='k')
    plt.ylabel(r'$\rho(r)/N_{type}$', fontsize=20,fontweight='normal', color='k')
    plt.legend()
    plt.gca().set_xlim(xmin=0,xmax=9)

    # Nome do arquivo dentro da pasta
    file_path_densidade = os.path.join(folder, "densidade_radial_" + str(indice) + '.png')

    # Salvar dentro da pasta
    plt.savefig(file_path_densidade, dpi=300)
    plt.close()  # Fecha a figura corretamente
    print(f"Imagem Hi-C salva em {file_path_densidade}")

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
            plt.savefig(file_output, dpi=300, bbox_inches="tight")
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

def analise_hi_c_densidade_dense(folder='resultado', arquivo_dense='resultado/hi_c_matrix_combinado.dense'):
    """Gera e salva a matriz Hi-C a partir de um arquivo .dense, além de um gráfico de distribuição radial."""

    # Criar a pasta se não existir
    os.makedirs(folder, exist_ok=True)

    # Nome dos arquivos dentro da pasta
    file_image = os.path.join(folder, f"hi_c_matrix_combinacao.png")
    file_radial = os.path.join(folder, f"densidade_radial_combinacao.png")

    # Carregar matriz Hi-C do arquivo .dense
    hic_matrix = np.loadtxt(arquivo_dense)

    # Verificar dimensões
    print(f"Matriz Hi-C carregada com formato: {hic_matrix.shape}")

    # Gerar e salvar o mapa Hi-C
    plt.matshow(hic_matrix, norm=mpl.colors.LogNorm(vmin=0.005, vmax=hic_matrix.max()), cmap="Reds")
    plt.colorbar()
    plt.savefig(file_image, dpi=300, bbox_inches="tight")
    print(f"Imagem Hi-C salva em {file_image}")

    # Necessário ajuste no código (adicinar para salvar dados .txt das distribuições radiais de cada simulação e posteior usar estes dados para gera um gráfico com as médias)
    # Gerar gráfico de distribuição radial baseado na matriz Hi-C
    radial_distribution = np.mean(hic_matrix, axis=1)  # Cálculo de média radial
    r = np.arange(len(radial_distribution))

    plt.figure(figsize=(10, 5))
    plt.plot(r, radial_distribution, color='blue', linestyle='-')
    plt.xlabel("Índice Radial", fontsize=14)
    plt.ylabel("Densidade Média", fontsize=14)
    plt.title("Distribuição Radial Hi-C")
    plt.grid(True)
    plt.savefig(file_radial, dpi=300)
    print(f"Gráfico de distribuição radial salvo em {file_radial}")

def notificar_som():
    winsound.Beep(970, 900)  # Toca um som de 970 Hz por 500ms

if __name__ == "__main__":
    # Sequência da cromatina
    nome_sequencia = 'chr10_beads.txt'
    pasta_resultado = 'resultado'
    criando_cromatina(nome_sequencia)

    # Quantidade de simulações
    replicas = 5
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
    analise_hi_c_densidade_dense()

    # Fim das simulações
    notificar_som()

contagem_final = time.time()  # Marca o fim
tempo = contagem_final - contagem_inicial  # Calcula a duração


print(f"Tempo de execução: {tempo:.2f} segundos")