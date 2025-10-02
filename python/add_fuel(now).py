import pygame
import sys
import random
import math # mathモジュールを追加

# --- 定数 ---
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
FPS = 60
NODE_RADIUS = 10
# LINE_WIDTH は画像を使用するため不要になる可能性が高いですが、残しておく
LINE_WIDTH = 3 
VEHICLE_SIZE = 40 # 車両のサイズを少し大きくして、アイコンを見やすくする
INITIAL_FUEL = 1000  # 初期燃料量

# 色の定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (150, 150, 150)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_GREY = (200, 200, 200)

# --- グラフ関連クラス (簡略化) ---
class Node:
    def __init__(self, id, x, y, node_type='city'):
        self.id = id
        self.x = x
        self.y = y
        self.type = node_type
        self.is_selected = False
        self.image = self.load_image_for_type(node_type)
        # スコア設定
        if node_type == 'customer':
            self.score = random.randint(50, 200)  # 顧客ノードは50-200点
        elif node_type == 'warehouse':
            self.score = 0  # 倉庫は0点
        else:  # city
            self.score = random.randint(20, 100)  # 都市ノードは20-100点
        self.visited = False  # 訪問済みフラグ

    def load_image_for_type(self, node_type):
        try:
            # 適切なアイコンのパスを設定してください
            if node_type == 'warehouse':
                return pygame.image.load('assets/warehouse_icon.png').convert_alpha()
            elif node_type == 'customer':
                return pygame.image.load('assets/customer_icon.png').convert_alpha()
            else: # 'city' or default
                return pygame.image.load('assets/city_icon.png').convert_alpha()
        except pygame.error:
            print(f"Warning: Could not load image for node type '{node_type}'. Using default circle.")
            return None

    def draw(self, screen):
        if self.image:
            img_rect = self.image.get_rect(center=(self.x, self.y))
            screen.blit(self.image, img_rect)
        else:
            color = BLUE if self.type == 'city' else RED if self.type == 'customer' else GREEN
            pygame.draw.circle(screen, color, (self.x, self.y), NODE_RADIUS)

        if self.is_selected:
            pygame.draw.circle(screen, YELLOW, (self.x, self.y), NODE_RADIUS + 8, 3)

        # 訪問済みのノードにはチェックマークを表示
        if self.visited:
            pygame.draw.circle(screen, GREEN, (self.x + NODE_RADIUS, self.y - NODE_RADIUS), 5)
            font_small = pygame.font.Font(None, 12)
            check_text = font_small.render("✓", True, WHITE)
            screen.blit(check_text, (self.x + NODE_RADIUS - 3, self.y - NODE_RADIUS - 4))

        font = pygame.font.Font(None, 20)
        text = font.render(str(self.id), True, BLACK)
        text_rect = text.get_rect(center=(self.x, self.y + NODE_RADIUS + 10))
        screen.blit(text, text_rect)
        
        # スコアを表示
        if self.score > 0:
            score_font = pygame.font.Font(None, 16)
            score_text = score_font.render(f"{self.score}pt", True, ORANGE)
            score_rect = score_text.get_rect(center=(self.x, self.y + NODE_RADIUS + 25))
            screen.blit(score_text, score_rect)

class Edge:
    def __init__(self, node1, node2, distance):
        self.node1 = node1
        self.node2 = node2
        self.distance = distance # この距離を時間や燃料コストに変換する
        self.is_highlighted = False # 新しく追加
        self.is_visible = True  # エッジが描画されるかどうかを制御

    def draw(self, screen):
        # 可視化されていないエッジは描画しない
        if not self.is_visible:
            return
            
        color = ORANGE if self.is_highlighted else LIGHT_GREY
        width = LINE_WIDTH + 2 if self.is_highlighted else LINE_WIDTH

        pygame.draw.line(screen, color, (self.node1.x, self.node1.y), (self.node2.x, self.node2.y), width)
        
        # 距離表示
        font = pygame.font.Font(None, 18)
        mid_x = (self.node1.x + self.node2.x) // 2
        mid_y = (self.node1.y + self.node2.y) // 2
        distance_text = font.render(f"{int(self.distance)}", True, BLACK)
        # エッジの中央から少しずらして表示
        screen.blit(distance_text, (mid_x + 5, mid_y + 5))


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.edge_map = {}

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, node1_id, node2_id, distance, edge_image=None): # edge_image 引数を追加
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)
        if node1 and node2:
            edge = Edge(node1, node2, distance) # image を渡す
            self.edges.append(edge)
            self.edge_map[(node1_id, node2_id)] = edge
            self.edge_map[(node2_id, node1_id)] = edge

    def get_edge_between(self, node1_id, node2_id):
        return self.edge_map.get((node1_id, node2_id)) or self.edge_map.get((node2_id, node1_id))

    def get_neighbors(self, node_id):
        neighbors = []
        for edge in self.edges:
            if edge.node1.id == node_id:
                neighbors.append(edge.node2.id)
            elif edge.node2.id == node_id:
                neighbors.append(edge.node1.id)
        return neighbors

    def find_shortest_path(self, start_node_id, end_node_id):
        distances = {node_id: float('infinity') for node_id in self.nodes}
        distances[start_node_id] = 0
        previous_nodes = {node_id: None for node_id in self.nodes}
        
        priority_queue = [(0, start_node_id)]

        while priority_queue:
            current_distance, current_node_id = min(priority_queue, key=lambda x: x[0])
            priority_queue.remove((current_distance, current_node_id))

            if current_node_id == end_node_id:
                path = []
                while previous_nodes[current_node_id] is not None:
                    path.insert(0, current_node_id)
                    current_node_id = previous_nodes[current_node_id]
                path.insert(0, start_node_id)
                return path

            for edge in self.edges:
                neighbor_id = None
                edge_distance = edge.distance
                if edge.node1.id == current_node_id:
                    neighbor_id = edge.node2.id
                elif edge.node2.id == current_node_id:
                    neighbor_id = edge.node1.id
                
                if neighbor_id:
                    new_distance = current_distance + edge_distance
                    if new_distance < distances[neighbor_id]:
                        distances[neighbor_id] = new_distance
                        previous_nodes[neighbor_id] = current_node_id
                        if (new_distance, neighbor_id) not in priority_queue:
                            priority_queue.append((new_distance, neighbor_id))
        return None

class Vehicle:
    def __init__(self, id, start_node, speed=5, vehicle_image=None): # vehicle_image 引数を追加
        self.id = id
        self.current_node = start_node
        self.x = start_node.x
        self.y = start_node.y
        self.speed = speed
        self.path_nodes_ids = []
        self.current_path_segment_start_id = None
        self.current_path_segment_end_id = None
        self.target_x, self.target_y = start_node.x, start_node.y
        self.is_moving = False
        self.path_index = 0
        self.current_highlighted_edge = None
        
        # 燃料システム
        self.fuel = INITIAL_FUEL
        self.max_fuel = INITIAL_FUEL
        self.out_of_fuel = False
        
        # スコアシステム
        self.score = 0
        self.selected_nodes = []  # 選択されたノードのリスト
        
        # 車両画像の設定
        self.original_image = vehicle_image
        self.base_image = None  # スケールされた基本画像
        self.rendered_image = None  # 回転後の画像
        self.angle = 0  # 現在の向き（度数法）
        
        if self.original_image:
            self.base_image = pygame.transform.scale(self.original_image, (VEHICLE_SIZE, VEHICLE_SIZE))
            self.rendered_image = self.base_image

    def set_path(self, node_id_list):
        # 燃料切れの場合は新しいパスを設定しない
        if self.out_of_fuel:
            print("燃料切れのため、新しいルートを設定できません。")
            return
        
        # 選択されたノードのリストを保存
        self.selected_nodes = node_id_list.copy()
            
        full_path = []
        if len(node_id_list) > 0:
            full_path.append(node_id_list[0])
            for i in range(len(node_id_list) - 1):
                segment_start = node_id_list[i]
                segment_end = node_id_list[i+1]
                shortest_segment_path = game_graph.find_shortest_path(segment_start, segment_end)
                if shortest_segment_path:
                    full_path.extend(shortest_segment_path[1:])
        
        # 燃料チェック：パス全体の距離を計算
        total_distance = 0
        for i in range(len(full_path) - 1):
            edge = game_graph.get_edge_between(full_path[i], full_path[i+1])
            if edge:
                total_distance += edge.distance
        
        # 燃料が足りない場合の警告
        if total_distance > self.fuel:
            print(f"警告: 燃料が不足しています。必要燃料: {int(total_distance)}, 現在燃料: {int(self.fuel)}")
            print("燃料が尽きるまで移動します。")
        
        self.path_nodes_ids = full_path
        self.path_index = 0
        
        if self.current_highlighted_edge:
            self.current_highlighted_edge.is_highlighted = False
            self.current_highlighted_edge = None

        if len(self.path_nodes_ids) > 1:
            self.set_next_target_segment()
        else:
            self.is_moving = False

    def set_next_target_segment(self):
        if self.path_index + 1 < len(self.path_nodes_ids):
            if self.current_highlighted_edge:
                self.current_highlighted_edge.is_highlighted = False

            self.current_path_segment_start_id = self.path_nodes_ids[self.path_index]
            self.current_path_segment_end_id = self.path_nodes_ids[self.path_index + 1]

            # 次のセグメントの燃料消費量をチェック
            next_edge = game_graph.get_edge_between(self.current_path_segment_start_id, self.current_path_segment_end_id)
            if next_edge:
                fuel_needed = next_edge.distance
                if self.fuel <= 0 and not self.out_of_fuel:
                    # 燃料が0以下になったが、まだ燃料切れ状態にしていない場合
                    # 現在のセグメントは完了させる
                    self.out_of_fuel = True
                    print(f"燃料切れ！現在のセグメントを完了してから停止します。")

            next_node = game_graph.nodes[self.current_path_segment_end_id]
            self.target_x, self.target_y = next_node.x, next_node.y
            self.is_moving = True

            self.current_highlighted_edge = game_graph.get_edge_between(self.current_path_segment_start_id, self.current_path_segment_end_id)
            # 可視化されているエッジのみハイライト
            if self.current_highlighted_edge and self.current_highlighted_edge.is_visible:
                self.current_highlighted_edge.is_highlighted = True
            else:
                self.current_highlighted_edge = None

            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = max(1, (dx**2 + dy**2)**0.5)
            self.move_vec_x = (dx / dist) * self.speed
            self.move_vec_y = (dy / dist) * self.speed
            
            # 移動方向の角度を計算（度数法）
            self.angle = math.degrees(math.atan2(-dy, dx))  # Pygameの座標系に合わせて-dyを使用
            self._update_rendered_image()
        else:
            self.is_moving = False
            if self.current_highlighted_edge:
                self.current_highlighted_edge.is_highlighted = False
                self.current_highlighted_edge = None

    def update(self):
        if not self.is_moving:
            return

        current_dist_to_target = ((self.target_x - self.x)**2 + (self.target_y - self.y)**2)**0.5

        if current_dist_to_target <= self.speed:
            # セグメント完了時の処理
            self.x, self.y = self.target_x, self.target_y
            arrived_node = game_graph.nodes[self.path_nodes_ids[self.path_index + 1]]
            self.current_node = arrived_node
            
            # スコア計算：選択されたノードに到達した場合
            if arrived_node.id in self.selected_nodes and not arrived_node.visited:
                self.score += arrived_node.score
                arrived_node.visited = True
                print(f"ノード {arrived_node.id} でスコア獲得: +{arrived_node.score}点 (総スコア: {self.score}点)")
            
            # 燃料消費
            if self.current_highlighted_edge:
                fuel_consumed = self.current_highlighted_edge.distance
                self.fuel -= fuel_consumed
                print(f"燃料消費: {int(fuel_consumed)}, 残り燃料: {int(max(0, self.fuel))}")
            
            self.path_index += 1
            
            # 燃料切れか次のセグメントがあるかチェック
            if self.out_of_fuel:
                # 燃料切れの場合は停止
                self.is_moving = False
                self.path_nodes_ids = []
                if self.current_highlighted_edge:
                    self.current_highlighted_edge.is_highlighted = False
                    self.current_highlighted_edge = None
                print(f"車両 {self.id} が燃料切れで停止しました。")
            elif self.path_index + 1 < len(self.path_nodes_ids):
                # 次のセグメントの燃料をチェック
                next_edge = game_graph.get_edge_between(self.path_nodes_ids[self.path_index], self.path_nodes_ids[self.path_index + 1])
                if next_edge and self.fuel < next_edge.distance:
                    # 次のセグメントに進む燃料がないが、現在のセグメントは完了させる
                    self.out_of_fuel = True
                    print(f"次のセグメントに進む燃料が不足しています。現在のノードで停止します。")
                    self.is_moving = False
                    self.path_nodes_ids = []
                    if self.current_highlighted_edge:
                        self.current_highlighted_edge.is_highlighted = False
                        self.current_highlighted_edge = None
                else:
                    self.set_next_target_segment()
            else:
                # パス完了
                self.is_moving = False
                self.path_nodes_ids = []
                if self.current_highlighted_edge:
                    self.current_highlighted_edge.is_highlighted = False
                    self.current_highlighted_edge = None
        else:
            self.x += self.move_vec_x
            self.y += self.move_vec_y

    def _update_rendered_image(self):
        """現在の角度に基づいて車両画像を回転させる"""
        if self.base_image:
            self.rendered_image = pygame.transform.rotate(self.base_image, self.angle)

    def draw(self, screen, offset_x=0, offset_y=0):
        # オフセットを適用した座標で描画
        draw_x = int(self.x + offset_x)
        draw_y = int(self.y + offset_y)
        
        # 燃料切れの場合は赤色で描画
        if self.out_of_fuel and self.rendered_image:
            # 赤いフィルターを適用
            red_surface = pygame.Surface(self.rendered_image.get_size())
            red_surface.set_alpha(100)
            red_surface.fill(RED)
            img_rect = self.rendered_image.get_rect(center=(draw_x, draw_y))
            screen.blit(self.rendered_image, img_rect)
            screen.blit(red_surface, img_rect)
        elif self.rendered_image:
            # 通常の描画
            img_rect = self.rendered_image.get_rect(center=(draw_x, draw_y))
            screen.blit(self.rendered_image, img_rect)
        else:
            # 画像がない場合のフォールバック
            color = RED if self.out_of_fuel else YELLOW
            pygame.draw.rect(screen, color, (draw_x - VEHICLE_SIZE // 2, draw_y - VEHICLE_SIZE // 2, VEHICLE_SIZE, VEHICLE_SIZE))
        
        font = pygame.font.Font(None, 18)
        text = font.render(str(self.id), True, BLACK)
        text_rect = text.get_rect(center=(draw_x, draw_y + VEHICLE_SIZE // 2 + 5)) # 車両の下にIDを表示
        screen.blit(text, text_rect)

# --- ゲーム本体 ---
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("物流ルート最適化ゲーム")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.game_state = 'PLANNING' # PLANNING, DELIVERING, RESULTS

        self.edge_image = None
        self.vehicle_image = None # 車両画像を保持する変数

        self.graph = Graph()
        self.nodes = {}
        self.vehicles = []
        self.selected_node = None
        self.route_planning_nodes = []
        
        # ランキング機能
        self.high_scores = []  # スコアを記録する配列（降順に保存）
        self.max_rankings = 10  # 最大10個のスコアを保存

        self.setup_game()

    def add_score_to_ranking(self, score):
        """スコアをランキングに追加する"""
        # スコアを配列に追加
        self.high_scores.append(score)
        
        # 降順でソート
        self.high_scores.sort(reverse=True)
        
        # 配列のサイズが上限を超える場合、最小値を削除
        if len(self.high_scores) > self.max_rankings:
            self.high_scores.pop()  # 最後の要素（最小値）を削除
        
        print(f"スコア {score} をランキングに追加しました。")

    def load_assets(self):
        try:
            self.vehicle_image = pygame.image.load('assets/vehicle_icon.png').convert_alpha()
            print("Vehicle image loaded successfully.")
        except pygame.error:
            print("Warning: Could not load 'assets/vehicle_icon.png'. Vehicles will be drawn as rectangles.")
            self.vehicle_image = None


    def setup_game(self):
        self.load_assets() # アセットの読み込みをここで行う

        # サンプルノードの追加
        self.graph.add_node(Node(1, 100, 100, 'warehouse'))
        self.graph.add_node(Node(2, 300, 150, 'customer'))
        self.graph.add_node(Node(3, 200, 300, 'customer'))
        self.graph.add_node(Node(4, 450, 250, 'city'))
        self.graph.add_node(Node(5, 600, 100, 'customer'))
        self.graph.add_node(Node(6, 700, 300, 'city'))
        self.graph.add_node(Node(7, 500, 400, 'customer'))
        self.graph.add_node(Node(8, 800, 150, 'customer'))
        self.graph.add_node(Node(9, 900, 400, 'customer'))
        self.graph.add_node(Node(10, 1050, 250, 'city'))

        # サンプルエッジの追加 (距離は適当に計算)
        node_ids = list(self.graph.nodes.keys())
        connections = set()
        all_edges = []  # 全てのエッジを一時的に保存
        
        # すべての可能なエッジを生成
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                n1_id = node_ids[i]
                n2_id = node_ids[j]
                node1 = self.graph.nodes[n1_id]
                node2 = self.graph.nodes[n2_id]
                distance = ((node1.x - node2.x)**2 + (node1.y - node2.y)**2)**0.5
                all_edges.append((n1_id, n2_id, distance))
        
        # 距離でソートして、短いエッジを優先
        all_edges.sort(key=lambda x: x[2])
        
        # 各ノードから最低2つのエッジを確保し、全体で適度な数に制限
        edge_count_per_node = {node_id: 0 for node_id in node_ids}
        selected_edges = []
        
        for n1_id, n2_id, distance in all_edges:
            # 各ノードが最低2つの接続を持つか、全体のエッジ数が制限以下の場合に追加
            if (edge_count_per_node[n1_id] < 3 or edge_count_per_node[n2_id] < 3) and len(selected_edges) < 15:
                selected_edges.append((n1_id, n2_id, distance))
                edge_count_per_node[n1_id] += 1
                edge_count_per_node[n2_id] += 1
        
        # 選択されたエッジをグラフに追加
        for n1_id, n2_id, distance in selected_edges:
            self.graph.add_edge(n1_id, n2_id, distance)

        warehouse_node = None
        for node in self.graph.nodes.values():
            if node.type == 'warehouse':
                warehouse_node = node
                break
        
        if warehouse_node:
            self.vehicles.append(Vehicle(101, warehouse_node, vehicle_image=self.vehicle_image)) # 画像を渡す
        else:
            print("Warning: No warehouse node found. Vehicle starting at node 1 (if exists).")
            if 1 in self.graph.nodes:
                self.vehicles.append(Vehicle(101, self.graph.nodes[1], vehicle_image=self.vehicle_image))
            else:
                print("Error: No nodes available to place vehicle.")

        global game_graph
        game_graph = self.graph

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.game_state = 'PLANNING'
            for node in self.graph.nodes.values():
                node.is_selected = False
                node.visited = False  # 訪問状態をリセット
                self.route_planning_nodes = []
                self.selected_node = None
                for edge in self.graph.edges:
                    edge.is_highlighted = False
                if self.vehicles:
                    warehouse_node = None
                    for node in self.graph.nodes.values():
                        if node.type == 'warehouse':
                            warehouse_node = node
                            break
                    if warehouse_node:
                        self.vehicles[0].current_node = warehouse_node
                        self.vehicles[0].x = warehouse_node.x
                        self.vehicles[0].y = warehouse_node.y
                        self.vehicles[0].is_moving = False                           
                        self.vehicles[0].path_nodes_ids = []
                        self.vehicles[0].path_index = 0
                        self.vehicles[0].current_highlighted_edge = None
                        # 燃料をリセット
                        self.vehicles[0].fuel = INITIAL_FUEL
                        self.vehicles[0].out_of_fuel = False
                        # スコアをリセット
                        self.vehicles[0].score = 0
                        self.vehicles[0].selected_nodes = []
                        print("ゲームリセット！プランニングモードに戻ります。")
                        
        if self.game_state == 'PLANNING':
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                node_clicked = False
                
                # オフセットを計算（描画時と同じ計算）
                if self.graph.nodes:
                    xs = [node.x for node in self.graph.nodes.values()]
                    ys = [node.y for node in self.graph.nodes.values()]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    graph_cx = (min_x + max_x) // 2
                    graph_cy = (min_y + max_y) // 2
                    screen_cx = SCREEN_WIDTH // 2
                    screen_cy = SCREEN_HEIGHT // 2
                    offset_x = screen_cx - graph_cx
                    offset_y = screen_cy - graph_cy
                else:
                    offset_x = 0
                    offset_y = 0
                
                for node_id, node in self.graph.nodes.items():
                    # オフセットを適用した座標で距離を計算
                    node_screen_x = node.x + offset_x
                    node_screen_y = node.y + offset_y
                    dist = ((mouse_pos[0] - node_screen_x)**2 + (mouse_pos[1] - node_screen_y)**2)**0.5
                    if dist <= NODE_RADIUS + 5:
                        if node.id in self.route_planning_nodes:
                            node.is_selected = False
                            self.route_planning_nodes.remove(node.id)
                            if self.selected_node and self.selected_node.id == node.id:
                                self.selected_node = None
                        else:
                            node.is_selected = True
                            self.route_planning_nodes.append(node.id)
                            self.selected_node = node
                        node_clicked = True
                        break
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if len(self.route_planning_nodes) >= 2:
                        self.vehicles[0].set_path(self.route_planning_nodes)
                        
                        for node_id in self.route_planning_nodes:
                            if node_id in self.graph.nodes:
                                self.graph.nodes[node_id].is_selected = False
                        self.route_planning_nodes = []
                        self.selected_node = None

                        self.game_state = 'DELIVERING'
                        print("配送開始！")
                    else:
                        print("配送ルートを2つ以上のノードで指定してください。")

        elif self.game_state == 'DELIVERING':
            pass

    def update(self):
        if self.game_state == 'DELIVERING':
            for vehicle in self.vehicles:
                vehicle.update()
            
            all_vehicles_idle = all(not v.is_moving for v in self.vehicles)
            if all_vehicles_idle:
                if any(v.out_of_fuel for v in self.vehicles):
                    print("燃料切れのため配送が中断されました。")
                else:
                    print("配送完了！")
                     # ゲーム終了時の燃料ボーナス計算
                    ##for vehicle in self.vehicles:
                    ##    fuel_saved = max(0, vehicle.fuel)
                    ##    fuel_bonus = int(fuel_saved)
                    ##    vehicle.score += fuel_bonus
                    ##   print(f"燃料ボーナス: +{fuel_bonus}点 (残り燃料: {int(fuel_saved)})")
                    ##    print(f"最終スコア: {vehicle.score}点")
                
                # スコアをランキングに追加
                if self.vehicles:
                    final_score = self.vehicles[0].score
                    self.add_score_to_ranking(final_score)
                
                self.game_state = 'RESULTS'

    def draw(self):
        self.screen.fill(WHITE)

        # --- グラフの中心を画面中央に合わせるためのオフセット計算 ---
        if self.graph.nodes:
            xs = [node.x for node in self.graph.nodes.values()]
            ys = [node.y for node in self.graph.nodes.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            graph_cx = (min_x + max_x) // 2
            graph_cy = (min_y + max_y) // 2
            screen_cx = SCREEN_WIDTH // 2
            screen_cy = SCREEN_HEIGHT // 2
            offset_x = screen_cx - graph_cx
            offset_y = screen_cy - graph_cy
        else:
            offset_x = 0
            offset_y = 0

        # --- ノード・エッジを一時的にずらして描画 ---
        # ノードの元座標を保存
        original_positions = {node_id: (node.x, node.y) for node_id, node in self.graph.nodes.items()}
        for node in self.graph.nodes.values():
            node.x += offset_x
            node.y += offset_y

        for edge in self.graph.edges:
            edge.draw(self.screen)
        for node_id, node in self.graph.nodes.items():
            node.draw(self.screen)
        if self.game_state == 'PLANNING' and len(self.route_planning_nodes) > 1:
            for i in range(len(self.route_planning_nodes) - 1):
                start_id = self.route_planning_nodes[i]
                end_id = self.route_planning_nodes[i+1]
                path_segment = self.graph.find_shortest_path(start_id, end_id)
                if path_segment and len(path_segment) > 1:
                    for j in range(len(path_segment) - 1):
                        node1 = self.graph.nodes[path_segment[j]]
                        node2 = self.graph.nodes[path_segment[j+1]]
                        pygame.draw.line(self.screen, BLUE, (node1.x, node1.y), (node2.x, node2.y), LINE_WIDTH + 2)
        
        # 車両を描画（オフセットを適用）
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, offset_x, offset_y)
            
        self.draw_ui()
        pygame.display.flip()

        # ノード座標を元に戻す
        for node_id, (ox, oy) in original_positions.items():
            self.graph.nodes[node_id].x = ox
            self.graph.nodes[node_id].y = oy

    def draw_ui(self):
        state_text = self.font.render(f"State: {self.game_state}", True, BLACK)
        self.screen.blit(state_text, (10, 10))
        
        # スコア表示
        if self.vehicles:
            vehicle = self.vehicles[0]
            score_text = self.font.render(f"Score: {vehicle.score}", True, BLACK)
            self.screen.blit(score_text, (10, 200))
        
        # ランキング表示（画面右上）
        self.draw_ranking()
        
        # 燃料情報を表示
        if self.vehicles:
            vehicle = self.vehicles[0]
            fuel_text = self.font.render(f"Fuel: {int(max(0, vehicle.fuel))}/{int(vehicle.max_fuel)}", True, BLACK)
            self.screen.blit(fuel_text, (10, 130))
            
            # 燃料バーの描画
            fuel_bar_width = 200
            fuel_bar_height = 20
            fuel_bar_x = 10
            fuel_bar_y = 160
            
            # 燃料バーの背景
            pygame.draw.rect(self.screen, LIGHT_GREY, (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height))
            
            # 燃料バーの前景
            fuel_ratio = max(0, vehicle.fuel) / vehicle.max_fuel
            fuel_bar_fill_width = int(fuel_bar_width * fuel_ratio)
            fuel_color = GREEN if fuel_ratio > 0.5 else ORANGE if fuel_ratio > 0.2 else RED
            pygame.draw.rect(self.screen, fuel_color, (fuel_bar_x, fuel_bar_y, fuel_bar_fill_width, fuel_bar_height))
            
            # 燃料バーの枠
            pygame.draw.rect(self.screen, BLACK, (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height), 2)
            
            # 燃料切れの警告
            if vehicle.out_of_fuel:
                warning_text = self.font.render("OUT OF FUEL!", True, RED)
                self.screen.blit(warning_text, (fuel_bar_x + fuel_bar_width + 20, fuel_bar_y - 5))

        if self.game_state == 'PLANNING':
            help_text_1 = self.font.render("Click on nodes to plan the route.", True, BLACK)
            help_text_2 = self.font.render("Press SPACE to start delivery. Press R to Reset.", True, BLACK)
            self.screen.blit(help_text_1, (10, 50))
            self.screen.blit(help_text_2, (10, 90))
        elif self.game_state == 'DELIVERING':
            delivering_text = self.font.render("Delivering... Please wait.", True, BLACK)
            self.screen.blit(delivering_text, (10, 50))
        elif self.game_state == 'RESULTS':
            if self.vehicles and not self.vehicles[0].out_of_fuel:
                self.game_state = 'PLANNING'
                return  # ここで早期リターンして描画をスキップ
            else:
                vehicle = self.vehicles[0] if self.vehicles else None
                final_score = vehicle.score if vehicle else 0
                results_text = self.font.render(f"Delivery complete! Final Score: {final_score}", True, GREEN)
                self.screen.blit(results_text, (SCREEN_WIDTH // 2 - results_text.get_width() // 2, SCREEN_HEIGHT // 2))
                reset_prompt = self.font.render("Press R to play again.", True, GREEN)
                self.screen.blit(reset_prompt, (SCREEN_WIDTH // 2 - reset_prompt.get_width() // 2, SCREEN_HEIGHT // 2 + 40))

    def draw_ranking(self):
        """ランキングを画面右上に表示する"""
        if not self.high_scores:
            return
        
        # ランキング表示の位置設定
        ranking_x = SCREEN_WIDTH - 250
        ranking_y = 10
        
        # タイトル表示
        ranking_font = pygame.font.Font(None, 28)
        title_text = ranking_font.render("HIGH SCORES", True, BLACK)
        self.screen.blit(title_text, (ranking_x, ranking_y))
        
        # ランキング背景の描画
        ranking_height = len(self.high_scores) * 25 + 40
        pygame.draw.rect(self.screen, (240, 240, 240), 
                        (ranking_x - 10, ranking_y - 5, 240, ranking_height), 0)
        pygame.draw.rect(self.screen, BLACK, 
                        (ranking_x - 10, ranking_y - 5, 240, ranking_height), 2)
        
        # 各スコアを表示
        score_font = pygame.font.Font(None, 24)
        for i, score in enumerate(self.high_scores):
            rank = i + 1
            score_text = score_font.render(f"{rank:2d}. {score:4d} pts", True, BLACK)
            y_pos = ranking_y + 30 + i * 25
            self.screen.blit(score_text, (ranking_x, y_pos))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_input(event)

            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    game = Game()
    game.run()