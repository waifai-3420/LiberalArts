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

        font = pygame.font.Font(None, 20)
        text = font.render(str(self.id), True, BLACK)
        text_rect = text.get_rect(center=(self.x, self.y + NODE_RADIUS + 10))
        screen.blit(text, text_rect)

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

# --- 車両クラス ---
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
        
        # 車両画像の設定
        self.original_image = vehicle_image
        self.base_image = None  # スケールされた基本画像
        self.rendered_image = None  # 回転後の画像
        self.angle = 0  # 現在の向き（度数法）
        
        if self.original_image:
            self.base_image = pygame.transform.scale(self.original_image, (VEHICLE_SIZE, VEHICLE_SIZE))
            self.rendered_image = self.base_image

    def set_path(self, node_id_list):
        full_path = []
        if len(node_id_list) > 0:
            full_path.append(node_id_list[0])
            for i in range(len(node_id_list) - 1):
                segment_start = node_id_list[i]
                segment_end = node_id_list[i+1]
                shortest_segment_path = game_graph.find_shortest_path(segment_start, segment_end)
                if shortest_segment_path:
                    full_path.extend(shortest_segment_path[1:])
        
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
            self.x, self.y = self.target_x, self.target_y
            self.current_node = game_graph.nodes[self.path_nodes_ids[self.path_index + 1]]
            self.path_index += 1
            
            if self.path_index + 1 < len(self.path_nodes_ids):
                self.set_next_target_segment()
            else:
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

    def draw(self, screen):
        if self.rendered_image:
            # 回転した画像の中心を車両の位置に合わせる
            img_rect = self.rendered_image.get_rect(center=(int(self.x), int(self.y)))
            screen.blit(self.rendered_image, img_rect)
        else:
            # 画像がない場合のフォールバック
            pygame.draw.rect(screen, YELLOW, (self.x - VEHICLE_SIZE // 2, self.y - VEHICLE_SIZE // 2, VEHICLE_SIZE, VEHICLE_SIZE))
        
        font = pygame.font.Font(None, 18)
        text = font.render(str(self.id), True, BLACK)
        text_rect = text.get_rect(center=(int(self.x), int(self.y) + VEHICLE_SIZE // 2 + 5)) # 車両の下にIDを表示
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

        self.setup_game()

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
                        print("ゲームリセット！プランニングモードに戻ります。")
        if self.game_state == 'PLANNING':
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                node_clicked = False
                for node_id, node in self.graph.nodes.items():
                    dist = ((mouse_pos[0] - node.x)**2 + (mouse_pos[1] - node.y)**2)**0.5
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
                self.game_state = 'RESULTS'
                print("Finalizing delivery!")

    def draw(self):
        self.screen.fill(WHITE)

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
                        pygame.draw.line(self.screen, BLUE, (node1.x, node1.y), (node2.x, node2.y), LINE_WIDTH + 2) # 太い青線で経路表示
        for vehicle in self.vehicles:
            vehicle.draw(self.screen)

        self.draw_ui()

        pygame.display.flip()

    def draw_ui(self):
        state_text = self.font.render(f"State: {self.game_state}", True, BLACK)
        self.screen.blit(state_text, (10, 10))

        if self.game_state == 'PLANNING':
            help_text_1 = self.font.render("Click on nodes to plan the route.", True, BLACK)
            help_text_2 = self.font.render("Press SPACE to start delivery. Press R to Reset.", True, BLACK)
            self.screen.blit(help_text_1, (10, 50))
            self.screen.blit(help_text_2, (10, 90))
        elif self.game_state == 'DELIVERING':
            delivering_text = self.font.render("Delivering... Please wait.", True, BLACK)
            self.screen.blit(delivering_text, (10, 50))
        elif self.game_state == 'RESULTS':
            results_text = self.font.render("Delivery complete! Score: (to be implemented)", True, BLACK)
            self.screen.blit(results_text, (SCREEN_WIDTH // 2 - results_text.get_width() // 2, SCREEN_HEIGHT // 2))
            reset_prompt = self.font.render("Press R to play again.", True, BLACK)
            self.screen.blit(reset_prompt, (SCREEN_WIDTH // 2 - reset_prompt.get_width() // 2, SCREEN_HEIGHT // 2 + 40))

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