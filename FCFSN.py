        avg_delay = sum(queue_delays) / len(queue_delays) if queue_delays else 0
        print(f"FCFSNoCooperation: {avg_delay:.2f}")
        print("\n=== Task Distribution ===")
        print(f"FCFSNoCooperation: Fog = {total_fog_count} ({total_fog_count / total_processed * 100:.1f}%), Cloud = {total_cloud_count} ({total_cloud_count / total_processed * 100:.1f}%)")
        print("\n=== Data Type Distribution ===")
        print(f"{'Data Type':<15} | {'Fog Count':<10} | {'Cloud Count':<10} | {'Total':<10} | {'Fog %':<10}")
        print("-" * 70)
        for data_type, counts in sorted(gateway.data_type_counts.items()):
            fog_type_count = counts.get('fog', 0)
            cloud_type_count = counts.get('cloud', 0)
            type_total = fog_type_count + cloud_type_count
            if type_total > 0:
                fog_type_pct = (fog_type_count / type_total) * 100
                print(f"{data_type:<15} | {fog_type_count:<10} | {cloud_type_count:<10} | {type_total:<10} | {fog_type_pct:<10.1f}%")
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
